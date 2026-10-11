"""Build manuscript artifacts only from completed, recorded experiments."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.camera_ready_protocol import read_config, repo_path, write_json


def export(config: dict) -> None:
    reported = json.loads((ROOT / 'configs/reported_results.json').read_text())
    rows = [{**row, 'provenance': 'reported'} for row in reported['rows']]
    uformer = {'model': 'Uformer-T', 'parameter_status': 'architecture_verified', 'provenance': 'newly_measured'}
    for looks in config['noise_looks']:
        directory = repo_path(config['output_dir']) / f'L{looks}'
        training = json.loads((directory / 'training_summary.json').read_text())
        results = json.loads((directory / 'evaluation_summary.json').read_text())
        if not training['training_finished']:
            raise ValueError('Refusing to export an unfinished/pilot training run as final results.')
        if training['signature'] != results['signature']:
            raise ValueError('Training/evaluation provenance does not match.')
        uformer.update(parameters=results['parameters'])
        uformer[f'L{looks}_psnr'] = results['primary_psnr']
        uformer[f'L{looks}_ssim'] = results['primary_ssim']
    rows.insert(len(rows) - 1, uformer)
    output = ROOT / 'paper/generated'; output.mkdir(parents=True, exist_ok=True)
    table = [r'\begin{table}[t]', r'\centering',
             r'\caption{FoE comparison at $L=1$ and $L=10$. Existing results are retained from the submitted experiments; Uformer-T is newly measured under the documented protocol.}',
             r'\label{tab:main_results}', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Method & Params & PSNR$_1$ & SSIM$_1$ & PSNR$_{10}$ & SSIM$_{10}$ \\', r'\midrule']
    for row in rows:
        params = r'NR$^\dagger$' if row['parameter_status'] == 'reported_count_unverified' else f'{row["parameters"]:,}'
        table.append(f'{row["model"]} & {params} & {row["L1_psnr"]:.2f} & {row["L1_ssim"]:.3f} & '
                     f'{row["L10_psnr"]:.2f} & {row["L10_ssim"]:.3f} ' + r'\\')
    table.extend([r'\bottomrule', r'\end{tabular}', r'\par\smallskip\footnotesize',
                  r'$^\dagger$The historical TNRD parameter count could not be verified against the supplied code. '
                  r'Original noisy test realizations were unavailable; historical and new scores are not paired measurements.',
                  r'\end{table}'])
    (output / 'results_table.tex').write_text('\n'.join(table) + '\n', encoding='utf-8')
    with (repo_path(config['output_dir']) / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    profile = json.loads((ROOT / 'results_camera_ready/complexity/complexity.json').read_text())
    if not profile['gpu']:
        raise ValueError('Final complexity export requires the allocated GPU benchmark, not CPU smoke timings.')
    gpu_label = str(profile['gpu']).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Architecture inference at batch 1, FP32 on '+gpu_label+r'. MACs count convolution, linear and attention matrix products; nonlinear operations are excluded.}',
             r'\label{tab:complexity}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Method & Size & Params & GMAC & Median ms \\', r'\midrule']
    for row in profile['rows']:
        label = row['model'].replace('Repository TNRD', 'Repo. TNRD')
        lines.append(f'{label} & {row["input_size"]} & {row["parameters"]:,} & '
                     f'{row["supported_gmacs"]:.3f} & {row["latency_median_ms"]:.2f} ' + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (output / 'complexity_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(output / 'provenance.json', {'comparison_rows': rows, 'complexity_environment': profile['environment'],
                                          'gpu': profile['gpu'], 'historical_note': reported['provenance']})
    # New Uformer qualitative panels use only the new saved corruption; no old
    # model outputs are presented as if they shared this observation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    destination = ROOT / 'paper/figures'; destination.mkdir(parents=True, exist_ok=True)
    for looks in config['noise_looks']:
        image_root = repo_path(config['output_dir']) / f'L{looks}/images'
        examples = sorted(image_root.iterdir())[:3]  # fixed selection, no metric ranking
        figure, axes = plt.subplots(len(examples), 3, figsize=(7.5, 2.5 * len(examples)), squeeze=False)
        for row, directory in enumerate(examples):
            for col, (name, title) in enumerate([('clean', 'Clean'), ('noisy', 'Noisy'), ('uformer', 'Uformer-T')]):
                axes[row, col].imshow(mpimg.imread(directory / f'{name}.png'), cmap='gray', vmin=0, vmax=1)
                axes[row, col].axis('off')
                if row == 0:
                    axes[row, col].set_title(title)
        figure.tight_layout()
        figure.savefig(destination / f'uformer_L{looks}.png', dpi=240)
        plt.close(figure)
    write_json(output / 'pipeline_complete.json', {'complete': True,
               'comparison_methods': len(rows), 'noise_looks': config['noise_looks'], 'gpu': profile['gpu']})
    print('Exported measured Uformer tables, provenance and qualitative panels.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/uformer_gamma.json')
    args = parser.parse_args()
    export(read_config(args.config))
