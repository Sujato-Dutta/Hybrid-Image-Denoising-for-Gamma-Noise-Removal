"""Replot submitted scientific panels, preserving their original image content.

Only layout changes. Pixel crops come from embedded figure assets extracted
from the author-supplied PDF; no model inference or synthesized detail is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def runs(mask: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    transitions = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.where(transitions == 1)[0], np.where(transitions == -1)[0])
            if b - a >= minimum]


def extract_grid(image: np.ndarray, columns: int) -> tuple[list[list[np.ndarray]], dict]:
    content = np.any(image[..., :3] < 240, axis=-1)
    ys = runs(content.mean(axis=1) > 0.30, 100)
    if len(ys) != 3:
        raise ValueError(f'Expected three submitted example rows, got {ys}')
    data = np.concatenate([content[a:b] for a, b in ys], axis=0)
    xs = runs(data.mean(axis=0) > 0.20, 100)
    if len(xs) != columns:
        raise ValueError(f'Expected {columns} submitted columns, got {xs}')
    return [[image[y0:y1, x0:x1] for x0, x1 in xs] for y0, y1 in ys], {'row_ranges': ys, 'column_ranges': xs}


def plot_grid(grid, order, labels, destination, title=None):
    figure, axes = plt.subplots(len(grid), len(order), figsize=(2.2 * len(order), 2.2 * len(grid)), squeeze=False)
    for row, panels in enumerate(grid):
        for col, index in enumerate(order):
            axes[row, col].imshow(panels[index], cmap='gray', interpolation='nearest', vmin=0, vmax=255)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(labels[col], fontsize=11)
    if title:
        figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    figure.savefig(destination, dpi=240, bbox_inches='tight')
    plt.close(figure)


def main():
    figures = ROOT / 'paper/figures'
    originals = ROOT / 'tmp/submitted_figures_original'
    originals.mkdir(parents=True, exist_ok=True)
    outputs = []
    sar_grid = None
    for name, count in [('fig_qualitative_foe_L1', 7), ('fig_qualitative_foe_L10', 7), ('fig_sar_comparison', 5)]:
        destination = figures / f'{name}.png'
        original = originals / destination.name
        if not original.exists():
            original.write_bytes(destination.read_bytes())
        image = np.asarray(Image.open(original).convert('RGB'))
        grid, bounds = extract_grid(image, count)
        if count == 7:
            order = [0, 1, 2, 3, 4, 6, 5]
            labels = ['Clean', 'Noisy', 'BM3D', 'PDE', 'TNRD', 'DnCNN', 'Hybrid++ (Ours)']
        else:
            sar_grid = grid
            order, labels = [0, 1, 2, 4, 3], ['Noisy SAR', 'PDE', 'TNRD', 'DnCNN', 'Hybrid++']
        plot_grid(grid, order, labels, destination)
        outputs.append({'figure': destination.name, 'source_order': order, **bounds})
    plot_grid([sar_grid[1]], [0, 4, 3], ['Noisy SAR', 'DnCNN', 'Hybrid++'],
              figures / 'fig_failure_case.png', title='Existing real-SAR example: saturation and detail loss')
    (ROOT / 'docs/SUBMITTED_FIGURE_LAYOUT.json').write_text(json.dumps(outputs, indent=2) + '\n')
    print('Replotted three submitted comparisons with Hybrid++ last; saved the existing SAR failure example.')


if __name__ == '__main__':
    main()
