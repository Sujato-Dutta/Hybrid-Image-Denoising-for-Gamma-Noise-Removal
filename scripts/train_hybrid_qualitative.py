"""Train the released Hybrid++ only for matched qualitative figures.

Uses train/validation partitions only. Never computes test benchmark scores or
changes historical accuracy tables. Every epoch and phase boundary is resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from run_uformer_gamma import capture_rng, restore_rng, save_checkpoint
from src.camera_ready_protocol import (CachedCorruptions, canonical_hash, environment_metadata,
    prepare_protocol, read_config, repo_path, write_json)
from src.dataset import GrayscaleImageDataset
from src.hybrid_plus_v2 import HybridPlusModelV2
from src.noise import add_gamma_noise
from src.utils import set_seed

OUTPUT = ROOT / 'results_camera_ready/hybrid_qualitative'
RECIPE = {'seed': 42, 'num_stages': 3, 'alpha': 1.0, 'beta': 2.0, 'batch_size': 4,
          'stage_epochs': 100, 'stage_lr': 0.001, 'scheduler_patience': 5,
          'scheduler_factor': 0.5, 'early_stopping_patience': 10,
          'finetune_epochs': 50, 'finetune_lr': 0.0001, 'finetune_min_lr': 0.000001,
          'looks': [1, 10], 'gradient_clip': 1.0, 'precision': 'float32',
          'validation': 'Fixed cached validation corruptions; average MSE across both looks',
          'purpose': 'New qualitative-only checkpoint; historical benchmark values retained'}


@torch.no_grad()
def validate(model, loaders, stage):
    model.eval()
    total, count = 0., 0
    for looks, loader in loaders.items():
        for batch in loader:
            target, noisy = batch['image'].cuda(), batch['noisy'].cuda()
            prediction = model(noisy, looks, upto_stage=stage)
            total += float(F.mse_loss(prediction, target).item())
            count += 1
    result = total / count
    if not torch.isfinite(torch.tensor(result)):
        raise FloatingPointError('Non-finite Hybrid++ validation loss')
    return result


def train(config, epoch_limit=None):
    assert torch.cuda.is_available(), 'A Slurm GPU allocation is required'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(RECIPE['seed'])
    manifest = prepare_protocol(config)
    sources = ['src/hybrid_plus_v2.py', 'src/pde_baseline.py', 'src/dataset.py',
               'src/noise.py', 'src/camera_ready_protocol.py', 'scripts/train_hybrid_qualitative.py']
    signature = canonical_hash({'recipe': RECIPE, 'protocol': manifest['protocol_hash'],
        'sources': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    complete = OUTPUT / 'training_complete.json'
    if complete.exists():
        if json.loads(complete.read_text())['signature'] != signature:
            raise ValueError('Completed qualitative checkpoint has a different source/protocol.')
        print('Hybrid++ qualitative training already complete.', flush=True)
        return
    model = HybridPlusModelV2(num_stages=3, alpha=1., beta=2.).cuda()
    assert sum(p.numel() for p in model.parameters()) == 44408
    train_paths = [repo_path(item['path']) for item in manifest['images']['train']]
    loader = DataLoader(GrayscaleImageDataset(train_paths, augment=True), batch_size=4,
                        shuffle=True, num_workers=0, pin_memory=True)
    val_loaders = {looks: DataLoader(CachedCorruptions(repo_path(config['protocol_dir']), 'val', looks),
                                  batch_size=1, num_workers=0) for looks in RECIPE['looks']}
    metadata = {'signature': signature, 'protocol_hash': manifest['protocol_hash'],
                'recipe': RECIPE, 'architecture_parameters': 44408,
                'environment': environment_metadata(), 'gpu': torch.cuda.get_device_name(0)}
    write_json(OUTPUT / 'run_metadata.json', metadata)
    last = OUTPUT / 'last.pt'
    saved = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
    if saved:
        if saved['signature'] != signature:
            raise ValueError('Qualitative checkpoint source/config/protocol mismatch.')
        model.load_state_dict(saved['model_state_dict'])
    phase_start = saved['phase_index'] if saved else 0
    history = saved['history'] if saved else []
    epoch_count = 0
    for phase in range(phase_start, 4):
        stage = phase + 1 if phase < 3 else 3
        for parameter in model.parameters():
            parameter.requires_grad = phase == 3
        if phase < 3:
            for module in (model.pde, model.level_embedder, model.stages[phase]):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=RECIPE['stage_lr'] if phase < 3 else RECIPE['finetune_lr'])
        scheduler = (ReduceLROnPlateau(optimizer, patience=5, factor=.5) if phase < 3 else
                     CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6))
        best, patience, first_epoch = float('inf'), 0, 1
        if saved and phase == saved['phase_index']:
            if saved['optimizer'] is not None:
                optimizer.load_state_dict(saved['optimizer'])
                scheduler.load_state_dict(saved['scheduler'])
                best, patience, first_epoch = saved['best_val'], saved['patience'], saved['phase_epoch']+1
            restore_rng(saved['rng'])
        label = f'stage{stage}' if phase < 3 else 'finetune'
        print(f'Hybrid++ phase={label} starting epoch={first_epoch}', flush=True)
        best_path = OUTPUT / f'{label}_best.pt'
        budget = 100 if phase < 3 else 50
        for epoch in range(first_epoch, budget+1):
            if phase < 3 and patience >= 10:
                break
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            total, count = 0., 0
            for batch in loader:
                clean = batch['image'].cuda()
                for looks in RECIPE['looks']:
                    optimizer.zero_grad(set_to_none=True)
                    noisy = add_gamma_noise(clean, looks)
                    loss = F.mse_loss(model(noisy, looks, upto_stage=stage), clean)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite Hybrid++ training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.item())
                    count += 1
            value = validate(model, val_loaders, stage)
            if phase < 3:
                scheduler.step(value)
            else:
                scheduler.step()
            if value < best:
                best, patience = value, 0
                save_checkpoint(best_path, {**metadata, 'model_state_dict': model.state_dict(),
                    'phase': label, 'epoch': epoch, 'best_val': best})
            else:
                patience += 1
            entry = {'phase': label, 'epoch': epoch, 'train_mse': total/count,
                     'val_mse': value, 'best_val': best, 'seconds': time.perf_counter()-started,
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            history.append(entry)
            save_checkpoint(last, {**metadata, 'model_state_dict': model.state_dict(),
                'phase_index': phase, 'phase_epoch': epoch, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'best_val': best, 'patience': patience,
                'history': history, 'rng': capture_rng()})
            write_json(OUTPUT / 'history.json', history)
            print(f'Hybrid++ {label} epoch={epoch}/{budget} train={entry["train_mse"]:.6f} '
                  f'val={value:.6f} seconds={entry["seconds"]:.1f}', flush=True)
            epoch_count += 1
            if epoch_limit is not None and epoch_count >= epoch_limit:
                return
        selected = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model_state_dict'])
        save_checkpoint(last, {**metadata, 'model_state_dict': model.state_dict(),
            'phase_index': phase+1, 'phase_epoch': 0, 'optimizer': None, 'scheduler': None,
            'best_val': best, 'patience': 0, 'history': history, 'rng': capture_rng()})
        saved = None
    save_checkpoint(OUTPUT / 'qualitative.pt', {**metadata, 'model_state_dict': model.state_dict(),
                    'selection': 'Best joint fine-tuning checkpoint by fixed-validation MSE'})
    write_json(complete, {**metadata, 'training_finished': True, 'epochs_completed': len(history),
                         'test_metrics_computed': False})
    print('Hybrid++ qualitative checkpoint ready; no test benchmark was run.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/uformer_gamma.json')
    parser.add_argument('--epoch-limit', type=int, help='Resumable resource pilot; not final training')
    args = parser.parse_args()
    train(read_config(args.config), args.epoch_limit)
