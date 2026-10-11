"""Measure actual FP32 training memory on the Slurm-assigned GPU before training."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.uformer import Uformer
from src.camera_ready_protocol import read_config, repo_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/uformer_gamma.json')
    args = parser.parse_args()
    config = read_config(args.config)
    assert torch.cuda.is_available(), 'Slurm-assigned CUDA device required'
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = Uformer(variant=config['variant'], in_channels=1, img_size=config['img_size'],
                    window_size=config['window_size'], drop_path_rate=config['drop_path_rate']).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    images = torch.rand(config['batch_size'], 1, 180, 180, device='cuda')
    target = torch.rand_like(images)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(images) - target).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    report = {'gpu': torch.cuda.get_device_name(0), 'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
              'batch_size': config['batch_size'], 'precision': 'float32', 'input_size': 180,
              'internal_size': config['img_size'], 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'two_step_seconds': time.perf_counter()-started,
              'finite_loss': bool(torch.isfinite(loss).item()),
              'purpose': 'Resource preflight only; random weights and inputs; discarded before real training.'}
    assert report['finite_loss']
    destination = repo_path(config['output_dir']) / 'gpu_preflight.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
