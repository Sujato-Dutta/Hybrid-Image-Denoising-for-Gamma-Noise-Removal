"""Deploy/run/fetch this experiment in a dedicated repository directory over SSH.

No private-key data is copied. Archives exclude .git, virtual environments,
temporary dependencies, other projects and preexisting results.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shlex
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_NAME = 'Hybrid-Image-Denoising-for-Gamma-Noise-Removal-cvip26'
DEFAULT_HOST = 'se23uari167@10.59.121.172'
DEFAULT_KEY = str(Path.home() / '.ssh/id_ed25519')


def ssh_args(args) -> list[str]:
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=15', '-i', args.key, args.host]


def remote_python(args, code: str, stdin: bytes | None = None, capture=False):
    command = 'python3 -c ' + shlex.quote(code)
    return subprocess.run(ssh_args(args) + [command], input=stdin, check=True,
                          stdout=subprocess.PIPE if capture else None)


def deploy(args) -> None:
    paths = []
    for directory in ('src', 'scripts', 'configs', 'tests', 'docs', 'paper', 'data/FoETrainingSets180'):
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.json', '.md', '.csv', '.slurm', '.tex', '.bib', '.png'):
                paths.append(path)
    paths.extend(ROOT.glob('*.py'))
    paths.append(ROOT / 'requirements-camera-ready.txt')
    # Reference model is included only for an independent verification command.
    reference = ROOT / 'tmp/references/Uformer/model.py'
    if reference.exists():
        paths.append(reference)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    payload = {'name': REMOTE_NAME, 'archive': base64.b64encode(buffer.getvalue()).decode(),
               'bundle_sha256': hashlib.sha256(buffer.getvalue()).hexdigest()}
    bootstrap = '''import sys,json,base64,io,zipfile,pathlib
p=json.load(sys.stdin);root=(pathlib.Path.home()/p['name']).resolve()
if root.parent!=pathlib.Path.home().resolve(): raise RuntimeError('Invalid workspace root')
marker=root/'.camera_ready_workspace'
if root.exists() and not marker.exists(): raise RuntimeError('Existing unowned directory: refusing deployment')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(p['archive']))) as z:
 for info in z.infolist():
  target=(root/info.filename).resolve()
  if not target.is_relative_to(root): raise RuntimeError('Archive path escaped workspace')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
marker.write_text(p['bundle_sha256'])
print(json.dumps({'remote_root':str(root),'bundle_sha256':p['bundle_sha256'],'files':len(z.infolist())}))
'''
    result = remote_python(args, bootstrap, json.dumps(payload).encode(), capture=True)
    metadata = json.loads(result.stdout)
    (ROOT / 'docs/REMOTE_WORKSPACE.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(metadata)


def run(args) -> None:
    root = f'$HOME/{REMOTE_NAME}'
    # Only explicitly selected fixed commands are exposed here; no shell strings
    # from config or remote output are executed.
    if args.step == 'setup':
        code = f'''import pathlib,subprocess
root=pathlib.Path.home()/{REMOTE_NAME!r}
subprocess.run(['python3','-m','venv','--system-site-packages',str(root/'.venv')],check=True)
python=str(root/'.venv/bin/python')
subprocess.run([python,'-m','pip','install','-r',str(root/'requirements-camera-ready.txt')],check=True,cwd=root)
subprocess.run([python,'-m','pip','freeze'],check=True,stdout=(root/'docs/remote_environment.lock.txt').open('w'))
# DGX rules prohibit compute workloads on the login node, including CPU tests.
allocation=['srun','--partition=cpu_student','--account=student','--qos=normal','--cpus-per-task=4','--mem=8G','--time=00:30:00']
subprocess.run(allocation+[python,'-m','unittest','discover','-s','tests','-v'],check=True,cwd=root)
subprocess.run(allocation+[python,'scripts/verify_uformer.py'],check=True,cwd=root)
subprocess.run(allocation+[python,'scripts/prepare_foe_protocol.py'],check=True,cwd=root)
'''
        remote_python(args, code)
    elif args.step == 'submit':
        remote_python(args, f'''import pathlib,subprocess,json
root=pathlib.Path.home()/{REMOTE_NAME!r};(root/'logs').mkdir(exist_ok=True)
result=subprocess.check_output(['sbatch','--parsable','--chdir='+str(root),'--export=ALL,HYBRID_REPO_ROOT='+str(root),str(root/'scripts/submit_uformer.slurm')],cwd=pathlib.Path.home(),text=True).strip()
(root/'docs/submitted_job.json').write_text(json.dumps({{'job_id':result,'script':'scripts/submit_uformer.slurm'}}))
print('Submitted Uformer job',result)
''')
    elif args.step == 'status':
        remote_python(args, f'''import pathlib,subprocess,json
root=pathlib.Path.home()/{REMOTE_NAME!r}
path=root/'docs/submitted_job.json'
if path.exists():
 job=json.loads(path.read_text())['job_id'].split(';')[0]
 subprocess.run(['squeue','-j',job,'--format=%i,%j,%T,%M,%R'])
 log=root/'logs'/('uformer-'+job+'.out')
 if log.exists(): print('\\n'.join(log.read_text(errors='replace').splitlines()[-20:]))
else: print('No experiment job submitted')
''')


def fetch(args) -> None:
    code = f'''import pathlib,zipfile,io,sys
root=pathlib.Path.home()/{REMOTE_NAME!r};buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
 for directory in ['results_camera_ready','paper/figures','paper/generated','logs']:
  for path in (root/directory).rglob('*'):
   if path.is_file(): z.write(path,path.relative_to(root).as_posix())
 for name in ['docs/submitted_job.json','docs/remote_environment.lock.txt','docs/UFORMER_FIDELITY_CHECK.json']:
  path=root/name
  if path.exists(): z.write(path,name)
sys.stdout.buffer.write(buffer.getvalue())
'''
    result = remote_python(args, code, capture=True)
    with zipfile.ZipFile(io.BytesIO(result.stdout)) as archive:
        for info in archive.infolist():
            path = (ROOT / info.filename).resolve()
            if not path.is_relative_to(ROOT):
                raise RuntimeError('Fetched archive path escaped the repository.')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(info))
    print('Fetched', len(archive.infolist()), 'experiment files under the repository.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('deploy', 'run', 'fetch'))
    parser.add_argument('--step', choices=('setup', 'submit', 'status'), default='status')
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--key', default=DEFAULT_KEY)
    args = parser.parse_args()
    {'deploy': deploy, 'run': run, 'fetch': fetch}[args.action](args)
