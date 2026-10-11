"""Build matched figures on exact Uformer cache inputs; no benchmark scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Springer requires scalable embedded fonts in the PDF figure exports.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from src.camera_ready_protocol import (CachedCorruptions, array_hash, prepare_protocol,
                                      read_config, repo_path, write_json)
from src.hybrid_plus_v2 import HybridPlusModelV2
from src.utils import save_image


def plot(rows, titles, path, row_labels=None):
    figure, axes = plt.subplots(len(rows), 4, figsize=(9, 2.25*len(rows)), squeeze=False)
    for i, arrays in enumerate(rows):
        for j, array in enumerate(arrays):
            axes[i,j].imshow(array, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            for spine in axes[i,j].spines.values():
                spine.set_visible(False)
            if i == 0:
                axes[i,j].set_title(titles[j])
        if row_labels:
            axes[i,0].set_ylabel(row_labels[i])
    figure.tight_layout()
    figure.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    figure.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/uformer_gamma.json')
    args = parser.parse_args()
    config = read_config(args.config)
    assert torch.cuda.is_available(), 'Allocated GPU required'
    torch.set_num_threads(4)
    checkpoint_path = ROOT / 'results_camera_ready/hybrid_qualitative/qualitative.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    manifest = prepare_protocol(config)
    assert checkpoint['protocol_hash'] == manifest['protocol_hash']
    model = HybridPlusModelV2(num_stages=3).cuda().eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    examples = sorted(Path(item['path']).stem for item in manifest['images']['test'])[:3]
    destination = ROOT / 'paper/figures'; destination.mkdir(parents=True, exist_ok=True)
    records, first_rows = [], []
    for looks in (1,10):
        dataset = CachedCorruptions(repo_path(config['protocol_dir']), 'test', looks, 0)
        positions = {Path(name).stem: i for i,name in enumerate(dataset.names)}
        rows = []
        for name in examples:
            batch = dataset[positions[name]]
            clean, noisy = batch['image'], batch['noisy']
            old_images = repo_path(config['output_dir']) / f'L{looks}/images/{name}'
            # Byte-for-byte display checks prove that the saved Uformer images
            # depict this exact clean image and noise realization.
            for label, tensor in [('clean', clean), ('noisy', noisy)]:
                expected = np.round(tensor.squeeze().numpy()*255).astype(np.uint8)
                actual = cv2.imread(str(old_images/f'{label}.png'), cv2.IMREAD_GRAYSCALE)
                if actual is None or not np.array_equal(actual, expected):
                    raise ValueError(f'Uformer {label} display does not match immutable cache: {name}, L={looks}')
            uformer = np.load(old_images/'restored.npy', allow_pickle=False).squeeze()
            hybrid = model(noisy.unsqueeze(0).cuda(), looks).cpu()
            if not torch.isfinite(hybrid).all():
                raise FloatingPointError('Non-finite Hybrid++ figure output')
            output = ROOT / f'results_camera_ready/hybrid_qualitative/images/L{looks}/{name}'
            output.mkdir(parents=True, exist_ok=True)
            np.save(output/'restored.npy', hybrid.numpy())
            save_image(output/'hybrid.png', hybrid)
            arrays = [clean.squeeze().numpy(), noisy.squeeze().numpy(), uformer, hybrid.squeeze().numpy()]
            rows.append(arrays)
            records.append({'name':name,'looks':looks,'replicate':0,
                'noisy_float32_sha256':array_hash(noisy.numpy()),
                'clean_float32_sha256':array_hash(clean.numpy()),
                'uformer_raw_sha256':hashlib.sha256((old_images/'restored.npy').read_bytes()).hexdigest(),
                'display_matches_uformer_clean_and_noisy':True})
        plot(rows, ['Clean', 'Noisy', 'Uformer-T', 'Hybrid++'], destination/f'hybrid_uformer_L{looks}')
        first_rows.append(rows[0])
    plot(first_rows, ['Clean', 'Noisy', 'Uformer-T', 'Hybrid++'],
         destination/'hybrid_uformer_paired', ['$L=1$', '$L=10$'])
    write_json(ROOT/'results_camera_ready/hybrid_qualitative/figure_provenance.json',
        {'complete':True,'hybrid_checkpoint':str(checkpoint_path.relative_to(ROOT)),
         'hybrid_checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
         'checkpoint_origin':'Separately trained qualitative-only Hybrid++; historical benchmark retained',
         'protocol_hash':manifest['protocol_hash'],'example_selection':'First three sorted fixed test names; combined figure uses first',
         'records':records,'panel_order':['Clean','Noisy','Uformer-T','Hybrid++'],
         'test_metrics_computed':False})
    print('Matched Hybrid++ / Uformer figures exported on identical cached noisy inputs.', flush=True)


if __name__ == '__main__':
    main()
