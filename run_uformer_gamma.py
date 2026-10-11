"""Train only native Uformer-T on the existing FoE Gamma benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.camera_ready_protocol import (
    REPO_ROOT, CachedCorruptions, canonical_hash, environment_metadata,
    prepare_protocol, read_config, repo_path, write_json,
)
from src.dataset import GrayscaleImageDataset
from src.noise import add_gamma_noise
from src.uformer import Uformer
from src.utils import set_seed


def build_model(config: dict) -> Uformer:
    return Uformer(variant=config['variant'], in_channels=1, img_size=config['img_size'],
                   window_size=config['window_size'], drop_path_rate=config['drop_path_rate'])


def experiment_signature(config: dict, manifest: dict, looks: int) -> str:
    stable_config = {k: v for k, v in config.items() if k not in ('epochs', 'output_dir', 'protocol_dir', 'data_root')}
    sources = ['src/uformer.py', 'src/noise.py', 'src/dataset.py', 'src/camera_ready_protocol.py', 'run_uformer_gamma.py']
    return canonical_hash({'config': stable_config, 'protocol': manifest['protocol_hash'], 'looks': looks,
                           'sources': {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in sources}})


def capture_rng() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def validation_mse(model: Uformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    squared_error, elements = 0.0, 0
    for batch in loader:
        clean, noisy = batch['image'].to(device), batch['noisy'].to(device)
        estimate = model(noisy)
        squared_error += float(F.mse_loss(estimate, clean, reduction='sum').item())
        elements += clean.numel()
    return squared_error / elements


def train(config: dict, looks: int, device: torch.device, max_epochs: int | None = None) -> dict:
    if looks not in config['noise_looks']:
        raise ValueError(f'Unsupported L={looks}')
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    effective_seed = config['model_seed'] + looks
    set_seed(effective_seed)
    manifest = prepare_protocol(config)
    signature = experiment_signature(config, manifest, looks)
    output = repo_path(config['output_dir']) / f'L{looks}'
    output.mkdir(parents=True, exist_ok=True)
    paths = [repo_path(entry['path']) for entry in manifest['images']['train']]
    train_loader = DataLoader(GrayscaleImageDataset(paths, augment=True),
                              batch_size=config['batch_size'], shuffle=True, num_workers=0,
                              pin_memory=device.type == 'cuda')
    val_loader = DataLoader(CachedCorruptions(repo_path(config['protocol_dir']), 'val', looks),
                           batch_size=1, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, patience=config['scheduler_patience'],
                                  factor=config['scheduler_factor'], min_lr=config['min_learning_rate'])
    best_val, best_epoch, patience, start_epoch, updates = float('inf'), 0, 0, 1, 0
    history = []
    last_path, best_path = output / 'last.pt', output / 'best.pt'
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('Checkpoint configuration, source or data differs; use a new output directory.')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        best_val, best_epoch, patience = saved['best_val'], saved['best_epoch'], saved['patience']
        start_epoch, updates = saved['epoch'] + 1, saved['updates']
        history = saved['history']
        restore_rng(saved['rng'])
        if not best_path.exists():
            raise FileNotFoundError('A resumed run requires its validation-selected best.pt.')
        print(f'Resuming L={looks} at epoch {start_epoch}', flush=True)
    epochs = config['epochs'] if max_epochs is None else max_epochs
    metadata = {
        'model': 'Uformer-T' if config['variant'] == 'tiny' else f'Uformer-{config["variant"]}',
        'variant': config['variant'], 'in_channels': 1, 'looks': looks,
        'architecture_parameters': sum(p.numel() for p in model.parameters()),
        'config': config, 'effective_training_seed': effective_seed,
        'signature': signature, 'protocol_hash': manifest['protocol_hash'],
        'environment': environment_metadata(), 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'training_output': 'unclipped global image residual', 'loss': 'MSE',
    }
    write_json(output / 'run_metadata.json', metadata)
    for epoch in range(start_epoch, epochs + 1):
        if patience >= config['early_stopping_patience']:
            break
        started = time.perf_counter()
        model.train()
        total_loss, count = 0.0, 0
        for batch in train_loader:
            clean = batch['image'].to(device)
            noisy = add_gamma_noise(clean, looks)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(noisy), clean)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total_loss += float(loss.item()) * clean.shape[0]
            count += clean.shape[0]
        val_loss = validation_mse(model, val_loader, device)
        if not np.isfinite(val_loss):
            raise FloatingPointError('Non-finite validation loss')
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val, best_epoch, patience = val_loss, epoch, 0
            save_checkpoint(best_path, {**metadata, 'model_state_dict': model.state_dict(),
                                       'epoch': epoch, 'best_val': best_val, 'updates': updates})
        else:
            patience += 1
        entry = {'epoch': epoch, 'train_mse': total_loss / count, 'val_mse': val_loss,
                 'lr': optimizer.param_groups[0]['lr'], 'seconds': time.perf_counter() - started,
                 'updates': updates, 'last_batch_gradient_norm': float(norm)}
        history.append(entry)
        save_checkpoint(last_path, {**metadata, 'model_state_dict': model.state_dict(),
                                   'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                                   'epoch': epoch, 'updates': updates, 'best_val': best_val,
                                   'best_epoch': best_epoch, 'patience': patience, 'history': history,
                                   'rng': capture_rng()})
        write_json(output / 'history.json', history)
        print(f'L={looks} epoch={epoch}/{epochs} train={entry["train_mse"]:.6f} '
              f'val={val_loss:.6f} best_epoch={best_epoch} seconds={entry["seconds"]:.1f}', flush=True)
    summary = {**metadata, 'completed_epochs': len(history), 'best_epoch': best_epoch,
               'best_validation_mse': best_val, 'updates': updates,
               'early_stopped': patience >= config['early_stopping_patience'],
               'requested_epochs': epochs,
               'training_finished': len(history) >= config['epochs'] or patience >= config['early_stopping_patience']}
    write_json(output / 'training_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/uformer_gamma.json')
    parser.add_argument('--looks', type=int, choices=(1, 10), required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--max-epochs', type=int, help='Pilot/explicit extension; resumes existing epoch state')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no allocated GPU is visible.')
    train(read_config(args.config), args.looks, device, args.max_epochs)


if __name__ == '__main__':
    main()
