"""Persisted FoE split and corruption protocol for the new baseline only."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import GrayscaleImageDataset, discover_images, split_dataset
from src.noise import add_gamma_noise

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str | Path) -> Path:
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f'Experiment paths must stay under the cloned repository: {value}')
    return resolved


def read_config(path: str | Path) -> dict:
    config = json.loads(repo_path(path).read_text(encoding='utf-8'))
    if config['noise_looks'] != [1, 10] or config['precision'] != 'float32':
        raise ValueError('The camera-ready protocol requires L=1/10 and float32.')
    if config['batch_size'] < 1 or config['epochs'] < 1 or config['test_replicates'] < 1:
        raise ValueError('Batch size, epochs and corruption replicates must be positive.')
    return config


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def environment_metadata() -> dict:
    packages = {}
    for name in ('torch', 'numpy', 'opencv-python', 'opencv-python-headless', 'scikit-image', 'matplotlib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'cuda_build': torch.version.cuda}


def prepare_protocol(config: dict) -> dict:
    data_root, output = repo_path(config['data_root']), repo_path(config['protocol_dir'])
    split = split_dataset(discover_images(data_root), seed=config['split_seed'])
    partitions = {name: getattr(split, name) for name in ('train', 'val', 'test')}
    source = {
        'split_seed': config['split_seed'], 'corruption_seed': config['corruption_seed'],
        'test_replicates': config['test_replicates'], 'looks': config['noise_looks'],
        'noise_protocol': config['noise_protocol'],
        'images': {name: [{'path': p.relative_to(REPO_ROOT).as_posix(),
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
                   for name, paths in partitions.items()},
    }
    fingerprint = canonical_hash(source)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_fingerprint'] != fingerprint:
            raise ValueError('Existing protocol differs from data/config. Choose a new protocol directory.')
        for name, payload in manifest['caches'].items():
            if not (output / name).is_file():
                raise FileNotFoundError(f'Missing immutable cache: {name}')
            with np.load(output / name, allow_pickle=False) as cache:
                for key, expected in payload['array_hashes'].items():
                    if array_hash(cache[key]) != expected:
                        raise ValueError(f'Cache checksum mismatch: {name}/{key}')
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    caches = {}
    for partition in ('val', 'test'):
        paths = partitions[partition]
        dataset = GrayscaleImageDataset(paths)
        clean = torch.stack([dataset[i]['image'] for i in range(len(dataset))])
        names = np.asarray([p.relative_to(REPO_ROOT).as_posix() for p in paths])
        replicates = 1 if partition == 'val' else config['test_replicates']
        for looks in config['noise_looks']:
            for replicate in range(replicates):
                noise_seed = int(canonical_hash([config['corruption_seed'], partition, looks, replicate])[:8], 16)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(noise_seed)
                    noisy = add_gamma_noise(clean, looks).numpy()
                filename = f'{partition}_L{looks}_r{replicate}.npz'
                arrays = {'clean': clean.numpy(), 'noisy': noisy, 'names': names}
                np.savez_compressed(output / filename, **arrays)
                caches[filename] = {'noise_seed': noise_seed,
                                    'array_hashes': {key: array_hash(value) for key, value in arrays.items()}}
    manifest = {**source, 'source_fingerprint': fingerprint, 'caches': caches,
                'generator_environment': environment_metadata()}
    manifest['protocol_hash'] = canonical_hash({'source': fingerprint, 'caches': caches})
    write_json(manifest_path, manifest)
    return manifest


class CachedCorruptions(Dataset):
    def __init__(self, protocol_dir: Path, split: str, looks: int, replicate: int = 0) -> None:
        with np.load(protocol_dir / f'{split}_L{looks}_r{replicate}.npz', allow_pickle=False) as cache:
            self.clean = torch.from_numpy(cache['clean'].copy())
            self.noisy = torch.from_numpy(cache['noisy'].copy())
            self.names = cache['names'].tolist()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        return {'image': self.clean[index], 'noisy': self.noisy[index], 'name': self.names[index]}
