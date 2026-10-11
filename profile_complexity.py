"""Profile architecture complexity without retraining existing denoisers."""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from run_uformer_gamma import build_model
from src.benchmark_models import DnCNN
from src.camera_ready_protocol import environment_metadata, read_config, repo_path, write_json
from src.hybrid_plus_v2 import DifferentiablePDE, HybridPlusModelV2
from src.pde_baseline import PDEConfig, nonlinear_smooth_diffusion_denoise
from src.tnrd_model import StagewiseTNRD


class ArithmeticCounter(TorchDispatchMode):
    """Counts convolution/linear/batched matmul MACs, including functional ops.

    Elementwise operations, normalization, reductions, activations, padding and
    FFT/block matching are excluded. This is not a total FLOP count.
    """
    def __init__(self) -> None:
        super().__init__()
        self.macs = Counter()
        self.other_ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        if name == 'aten.conv2d.default':
            image, weight = args[:2]
            groups = args[6] if len(args) > 6 else 1
            self.macs['convolution'] += result.numel() * (image.shape[1] // groups) * weight.shape[-2] * weight.shape[-1]
        elif name == 'aten.conv_transpose2d.input':
            image, weight = args[:2]
            self.macs['convolution'] += image.numel() * weight.shape[1] * weight.shape[-2] * weight.shape[-1]
        elif name == 'aten.linear.default':
            self.macs['matrix_product'] += result.numel() * args[1].shape[-1]
        elif name == 'aten.matmul.default':
            self.macs['matrix_product'] += result.numel() * args[0].shape[-1]
        elif name == 'aten.convolution.default':
            image, weight = args[:2]
            transposed, groups = args[6], args[8]
            kernel = weight.shape[-2] * weight.shape[-1]
            if transposed:
                macs = image.numel() * weight.shape[1] * kernel
            else:
                macs = result.numel() * (image.shape[1] // groups) * kernel
            self.macs['convolution'] += macs
        elif name in ('aten.mm.default', 'aten.bmm.default'):
            self.macs['matrix_product'] += result.numel() * args[0].shape[-1]
        elif name == 'aten.addmm.default':
            self.macs['matrix_product'] += result.numel() * args[1].shape[-1]
        else:
            self.other_ops[name] += 1
        return result


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def profile(callable_model, parameters, image, device, warmups, repeats) -> dict:
    for _ in range(warmups):
        callable_model(image)
    synchronize(device)
    milliseconds = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        base_memory = torch.cuda.memory_allocated(device)
    else:
        base_memory = None
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        callable_model(image)
        synchronize(device)
        milliseconds.append(1000 * (time.perf_counter() - start))
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    counter = ArithmeticCounter()
    with counter:
        callable_model(image)
    total = sum(counter.macs.values())
    if total == 0:
        raise RuntimeError('Profiler did not capture arithmetic operations; refusing to report zero cost.')
    return {'parameters': parameters, 'supported_macs': total, 'supported_gmacs': total / 1e9,
            'supported_gflops_2_per_mac': 2 * total / 1e9,
            'latency_median_ms': float(np.median(milliseconds)),
            'latency_p95_ms': float(np.percentile(milliseconds, 95)),
            'base_allocated_bytes': base_memory, 'peak_allocated_bytes': peak,
            'additional_peak_bytes': peak - base_memory if peak is not None else None,
            'mac_breakdown': dict(counter.macs), 'excluded_operations': dict(counter.other_ops)}


def run(config: dict, device: torch.device, sizes: list[int], warmups: int, repeats: int) -> dict:
    if repeats < 1 or warmups < 0:
        raise ValueError('Invalid benchmark iteration counts.')
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026)
    output = repo_path('results_camera_ready/complexity')
    rows, details = [], []
    builders = [
        ('PDE baseline', None),
        ('Hybrid++', lambda: HybridPlusModelV2(num_stages=3)),
        ('DnCNN', lambda: DnCNN()),
        ('Repository TNRD', lambda: StagewiseTNRD(num_stages=7, learnable_filters=False)),
        ('Uformer-T', lambda: build_model(config)),
    ]
    for name, builder in builders:
        model = builder().to(device).eval() if builder else None
        provenance = 'initialized_architecture'
        if name == 'Uformer-T':
            checkpoint_path = repo_path(config['output_dir']) / 'L1' / 'best.pt'
            if checkpoint_path.exists():
                saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model_state_dict'])
                provenance = 'trained_L1_checkpoint'
        parameters = sum(p.numel() for p in model.parameters()) if model else 0
        function = (lambda x: nonlinear_smooth_diffusion_denoise(x, 1, PDEConfig())) if model is None else (
            (lambda x: model(x, 1)) if name in ('Hybrid++', 'Repository TNRD') else model
        )
        for size in sizes:
            image = torch.rand(1, 1, size, size, device=device)
            result = profile(function, parameters, image, device, warmups, repeats)
            details.append({'model': name, 'input_size': size, **result})
            row = {k: v for k, v in result.items() if k not in ('mac_breakdown', 'excluded_operations')}
            row.update({'model': name, 'input_size': size,
                        'padded_input_size': model.padded_side(size, size) if name == 'Uformer-T' else size,
                        'batch': 1, 'precision': 'float32', 'device': str(device), 'weight_provenance': provenance})
            rows.append(row)
            print(name, size, 'params', parameters, 'GMAC', f'{result["supported_gmacs"]:.3f}',
                  'median_ms', f'{result["latency_median_ms"]:.2f}', flush=True)
            del image
        del function, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'complexity.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {'environment': environment_metadata(), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
               'warmups': warmups, 'repeats': repeats, 'rows': rows, 'operator_details': details,
               'counting_convention': 'Convolution/linear/attention matrix products only; one MAC=two FLOPs; other operations excluded.',
               'timing_boundary': 'Synchronized inference call with input already on device, excluding disk and transfers.'}
    write_json(output / 'complexity.json', payload)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/uformer_gamma.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--sizes', type=int, nargs='+', default=[180, 256])
    parser.add_argument('--warmups', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    run(read_config(args.config), torch.device(args.device), args.sizes, args.warmups, args.repeats)
