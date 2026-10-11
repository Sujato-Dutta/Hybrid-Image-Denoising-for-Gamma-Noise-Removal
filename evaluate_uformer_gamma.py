"""Evaluate new Uformer checkpoints; historical Hybrid++ metrics stay separate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from run_uformer_gamma import build_model, experiment_signature
from src.camera_ready_protocol import CachedCorruptions, prepare_protocol, read_config, repo_path, write_json
from src.metrics import compute_psnr, compute_ssim
from src.noise import EPS
from src.utils import save_image


@torch.no_grad()
def evaluate(config: dict, device: torch.device, replicates: int = 1) -> list[dict]:
    torch.set_num_threads(config['cpu_threads'])
    if not 1 <= replicates <= config['test_replicates']:
        raise ValueError('Requested replicate count is not in the prepared protocol.')
    manifest = prepare_protocol(config)
    summaries = []
    for looks in config['noise_looks']:
        output = repo_path(config['output_dir']) / f'L{looks}'
        checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
        if checkpoint['signature'] != experiment_signature(config, manifest, looks):
            raise ValueError('Evaluation data/config/source differs from checkpoint.')
        model = build_model(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        rows = []
        for replicate in range(replicates):
            dataset = CachedCorruptions(repo_path(config['protocol_dir']), 'test', looks, replicate)
            for batch in DataLoader(dataset, batch_size=1, shuffle=False):
                clean, noisy = batch['image'].to(device), batch['noisy'].to(device)
                estimate = model(noisy).clamp(EPS, 1.0)
                name = batch['name'][0]
                rows.append({'image': name, 'looks': looks, 'replicate': replicate,
                             'model': 'Uformer-T', 'psnr': compute_psnr(clean, estimate),
                             'ssim': compute_ssim(clean, estimate), 'provenance': 'newly_measured'})
                if replicate == 0:
                    directory = output / 'images' / Path(name).stem
                    directory.mkdir(parents=True, exist_ok=True)
                    np.save(directory / 'restored.npy', estimate.cpu().numpy())
                    for label, tensor in [('clean', clean), ('noisy', noisy), ('uformer', estimate)]:
                        save_image(directory / f'{label}.png', tensor)
        with (output / 'test_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        primary = [r for r in rows if r['replicate'] == 0]
        by_replicate = [{'replicate': r, 'psnr_mean': float(np.mean([x['psnr'] for x in rows if x['replicate'] == r])),
                         'ssim_mean': float(np.mean([x['ssim'] for x in rows if x['replicate'] == r]))}
                        for r in range(replicates)]
        summary = {'looks': looks, 'model': 'Uformer-T', 'provenance': 'newly_measured',
                   'num_test_images': len(primary), 'replicates': replicates,
                   'primary_psnr': float(np.mean([r['psnr'] for r in primary])),
                   'primary_ssim': float(np.mean([r['ssim'] for r in primary])),
                   'image_psnr_std': float(np.std([r['psnr'] for r in primary])),
                   'image_ssim_std': float(np.std([r['ssim'] for r in primary])),
                   'by_replicate': by_replicate, 'checkpoint_epoch': checkpoint['epoch'],
                   'signature': checkpoint['signature'], 'protocol_hash': manifest['protocol_hash'],
                   'parameters': checkpoint['architecture_parameters']}
        write_json(output / 'evaluation_summary.json', summary)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(repo_path(config['output_dir']) / 'evaluation_summary.json', summaries)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/uformer_gamma.json')
    parser.add_argument('--replicates', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    evaluate(read_config(args.config), torch.device(args.device), args.replicates)
