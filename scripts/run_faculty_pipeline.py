"""Checkpoint continuation inside Slurm; queue at most one dependent successor.

Only timeout warnings cause automatic continuation. Ordinary errors stop the
chain for investigation. No model/training source is modified by this wrapper.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    workflow = os.environ.get('HYBRID_WORKFLOW', 'uformer')
    if workflow not in ('uformer', 'qualitative'):
        raise ValueError('Unknown experiment workflow')
    complete = ROOT / ('results_camera_ready/hybrid_qualitative/figure_provenance.json'
                       if workflow == 'qualitative' else 'paper/generated/pipeline_complete.json')
    # Written only after all export steps finish successfully.
    if complete.exists():
        print('Completed manuscript artifacts already exist; continuation exits.', flush=True)
        return 0
    warning = False
    pending_job = None
    iteration = int(os.environ.get('HYBRID_CHAIN_ITER', '1'))
    current = os.environ['SLURM_JOB_ID']
    def warned(signum, frame):
        nonlocal warning
        warning = True
    signal.signal(signal.SIGUSR1, warned)
    def cancel_successor():
        if pending_job:
            subprocess.run(['scancel', pending_job], check=True)
    commands = [
        ['scripts/prepare_foe_protocol.py', '--config', 'configs/uformer_gamma.json'],
        ['run_uformer_gamma.py', '--config', 'configs/uformer_gamma.json', '--looks', '1', '--device', 'cuda'],
        ['run_uformer_gamma.py', '--config', 'configs/uformer_gamma.json', '--looks', '10', '--device', 'cuda'],
        ['evaluate_uformer_gamma.py', '--config', 'configs/uformer_gamma.json', '--device', 'cuda', '--replicates', '5'],
        ['profile_complexity.py', '--config', 'configs/uformer_gamma.json', '--device', 'cuda'],
        ['scripts/export_camera_ready_results.py', '--config', 'configs/uformer_gamma.json'],
    ]
    if workflow == 'qualitative':
        commands = [['scripts/train_hybrid_qualitative.py'], ['scripts/plot_hybrid_uformer_paired.py']]
    print(f'Resuming pipeline in job {current}, allocation {iteration}/16.', flush=True)
    for arguments in commands:
        child = subprocess.Popen([sys.executable, *arguments], cwd=ROOT)
        while child.poll() is None:
            if warning and pending_job is None and iteration < 16:
                export = f'ALL,HYBRID_REPO_ROOT={ROOT},HYBRID_CHAIN_ITER={iteration+1},HYBRID_WORKFLOW={workflow}'
                script = 'submit_hybrid_qualitative.slurm' if workflow == 'qualitative' else 'resume_uformer_faculty.slurm'
                output = subprocess.check_output([
                    'sbatch', '--parsable', f'--dependency=afterany:{current}',
                    f'--chdir={ROOT}', f'--export={export}',
                    str(ROOT / 'scripts' / script),
                ], cwd=Path.home(), text=True).strip()
                pending_job = output.split(';')[0]
                if not pending_job.isdigit():
                    raise RuntimeError('Slurm continuation did not return a numeric job ID.')
                metadata = {'job_id': pending_job, 'predecessor': current, 'allocation': iteration+1,
                            'dependency': 'afterany', 'purpose': 'Checkpoint continuation after timeout warning'}
                path = ROOT / ('docs/hybrid_qualitative_job.json' if workflow == 'qualitative'
                               else 'docs/faculty_continuation_job.json')
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(metadata, indent=2)+'\n')
                temporary.replace(path)
                print(f'Timeout warning: successor {pending_job} waits for job {current} to end.', flush=True)
            time.sleep(1)
        if child.returncode:
            cancel_successor()
            print('Pipeline error; automatic continuation stopped for investigation.', flush=True)
            return child.returncode
    cancel_successor()
    print(f'{workflow} pipeline completed.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
