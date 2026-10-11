"""Faculty-account SSH management; .env is read locally and never transferred.

Only Slurm and file-management commands run on the login node. Scientific code
and environment preparation belong in allocated batch jobs.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import shlex
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp/ssh-deps'))
import paramiko

HOST = '10.59.121.172'
USER = 'muef2038'
NAME = 'Hybrid-Image-Denoising-for-Gamma-Noise-Removal-cvip26'


def password() -> str:
    raw = (ROOT / '.env').read_text(encoding='utf-8-sig').rstrip('\r\n')
    lines = [line for line in raw.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    for line in lines:
        key, separator, value = line.partition('=')
        if separator and ('PASSWORD' in key.upper() or key.upper().strip() in ('PASS', 'DGX_PASS')):
            return value.strip().strip('\"\'')
    if len(lines) == 1:
        return lines[0]
    raise ValueError('Cannot identify the local SSH password; file contents are suppressed.')


def connect():
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home() / '.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(HOST, username=USER, password=password(), look_for_keys=False,
                       allow_agent=False, timeout=20, auth_timeout=20, banner_timeout=20)
    except Exception as error:
        client.close()
        raise RuntimeError(f'SSH connection failed ({type(error).__name__}); credentials suppressed.') from None
    return client


def command(client, value, check=True):
    _, stdout, stderr = client.exec_command(value)
    output = stdout.read().decode(errors='replace')
    errors = stderr.read().decode(errors='replace')
    status = stdout.channel.recv_exit_status()
    if output:
        print(output.rstrip())
    if errors:
        print(errors.rstrip(), file=sys.stderr)
    if check and status:
        raise RuntimeError(f'Remote management command exited {status}.')
    return output


def mkdirs(sftp, path):
    try:
        sftp.stat(path)
    except FileNotFoundError:
        parent = posixpath.dirname(path)
        if parent != path:
            mkdirs(sftp, parent)
        sftp.mkdir(path)


def deploy(client):
    with client.open_sftp() as sftp:
        home = sftp.normalize('.')
        root = posixpath.join(home, NAME)
        marker = posixpath.join(root, '.camera_ready_workspace')
        try:
            sftp.stat(root)
        except FileNotFoundError:
            sftp.mkdir(root)
            with sftp.file(marker, 'w') as stream:
                stream.write('Hybrid++ CVIP26 faculty workspace\n')
        else:
            sftp.stat(marker)  # Do not overwrite an unrelated existing directory.
        files = []
        for folder in ('src', 'scripts', 'configs', 'tests', 'docs', 'data/FoETrainingSets180'):
            files.extend(path for path in (ROOT / folder).rglob('*')
                         if path.is_file() and '__pycache__' not in path.parts
                         and path.suffix in ('.py', '.json', '.md', '.csv', '.slurm', '.png'))
        files.extend(ROOT.glob('*.py'))
        files.append(ROOT / 'requirements-camera-ready.txt')
        reference = ROOT / 'tmp/references/Uformer/model.py'
        if reference.exists():
            files.append(reference)
        for path in sorted(set(files)):
            destination = posixpath.join(root, path.relative_to(ROOT).as_posix())
            mkdirs(sftp, posixpath.dirname(destination))
            sftp.put(str(path), destination)
        for folder in ('logs', 'paper/generated'):
            mkdirs(sftp, posixpath.join(root, folder))
        metadata = {'host': HOST, 'username': USER, 'remote_root': root, 'files': len(set(files)),
                    'credential_transferred': False}
        (ROOT / 'docs/REMOTE_WORKSPACE_FACULTY.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(json.dumps(metadata))


def fetch(client, metadata_only=False, manuscript_only=False):
    count = 0
    with client.open_sftp() as sftp:
        root = posixpath.join(sftp.normalize('.'), NAME)
        def download(relative):
            nonlocal count
            remote = posixpath.join(root, relative)
            try:
                entries = sftp.listdir_attr(remote)
            except FileNotFoundError:
                return
            for entry in entries:
                child = posixpath.join(relative, entry.filename)
                destination = (ROOT / child).resolve()
                if not destination.is_relative_to(ROOT):
                    raise ValueError('Remote artifact escaped the repository.')
                if stat.S_ISDIR(entry.st_mode):
                    download(child)
                elif stat.S_ISREG(entry.st_mode):
                    if metadata_only and destination.suffix not in ('.json', '.csv', '.out', '.log', '.txt', '.tex'):
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(posixpath.join(root, child), str(destination))
                    count += 1
        folders = ('paper/figures', 'paper/generated') if manuscript_only else (
            'results_camera_ready', 'paper/figures', 'paper/generated', 'logs')
        for folder in folders:
            download(folder)
        for name in ('docs/faculty_environment.lock.txt', 'docs/UFORMER_FIDELITY_CHECK.json',
                     'docs/faculty_continuation_job.json', 'docs/hybrid_qualitative_job.json'):
            try:
                sftp.get(posixpath.join(root, name), str(ROOT / name))
                count += 1
            except FileNotFoundError:
                pass
    print('Fetched', count, 'project artifacts via SFTP.')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'deploy', 'sync', 'fetch', 'status', 'submit'))
    parser.add_argument('--script', default='scripts/submit_uformer_faculty.slurm')
    parser.add_argument('--job', type=int)
    parser.add_argument('--dependency', type=int, help='Require successful completion of this setup job.')
    parser.add_argument('--afterany', type=int, help='Resume only after this GPU allocation has ended.')
    parser.add_argument('--file', action='append', help='Repository script to sync without redeploying training source.')
    parser.add_argument('--metadata-only', action='store_true', help='Fetch logs and JSON/CSV metadata without checkpoints or images.')
    parser.add_argument('--manuscript-only', action='store_true', help='Fetch generated manuscript tables and figures without training checkpoints.')
    args = parser.parse_args()
    with connect() as client:
        if args.action == 'inspect':
            for value in ('date -Is; hostname; id; pwd',
                          'sacctmgr -nP show assoc where user=muef2038 format=Account,Partition,QOS,DefaultQOS,MaxJobs,MaxSubmitJobs,MaxTRES',
                          'scontrol show partition cpu_faculty; scontrol show partition gpu_faculty',
                          'sinfo -p gpu_faculty --format="%P %a %l %D %t %G"',
                          'squeue -u muef2038 --format="%i,%j,%T,%M,%R"',
                          'sacctmgr -nP show qos normal,cpu_faculty,gpu_faculty format=Name,MaxTRESPerUser,MaxTRESPerJob,MaxWall,MaxJobsPerUser',
                          'scontrol show nodes | egrep "NodeName=|Gres=|CfgTRES=|AllocTRES="',
                          'command -v python3; command -v conda; ls -d /opt/conda* /opt/pytorch* /usr/local/bin/python* 2>/dev/null'):
                command(client, value, check=False)
        elif args.action == 'deploy':
            deploy(client)
        elif args.action == 'fetch':
            fetch(client, args.metadata_only, args.manuscript_only)
        elif args.action == 'sync':
            with client.open_sftp() as sftp:
                root = posixpath.join(sftp.normalize('.'), NAME)
                sftp.stat(posixpath.join(root, '.camera_ready_workspace'))
                for name in args.file or []:
                    path = (ROOT / name).resolve()
                    if not path.is_relative_to(ROOT / 'scripts') or path.suffix not in ('.py', '.slurm'):
                        raise ValueError('Only repository scripts may be synced.')
                    destination = posixpath.join(root, path.relative_to(ROOT).as_posix())
                    sftp.put(str(path), destination)
                    print('Synced', path.relative_to(ROOT).as_posix())
        elif args.action == 'status':
            queue = command(client, 'date -Is; squeue -u muef2038 --format="%i,%j,%T,%M,%R"')
            if not args.job:
                for line in queue.splitlines():
                    fields = line.split(',')
                    if len(fields) == 5 and fields[0].isdigit() and fields[1].startswith('hybrid-cvip') and fields[2] == 'RUNNING':
                        args.job = int(fields[0])
                        break
            if args.job:
                command(client, f'sacct -j {args.job} --format=JobID,JobName,State,Elapsed,ExitCode; '
                        f'scontrol show job {args.job}', check=False)
                with client.open_sftp() as sftp:
                    found = False
                    for prefix in ('uformer', 'setup', 'qualitative'):
                        path = posixpath.join(sftp.normalize('.'), NAME, f'logs/{prefix}-{args.job}.out')
                        try:
                            with sftp.file(path) as stream:
                                print('\n'.join(stream.read().decode(errors='replace').splitlines()[-35:]))
                            found = True
                        except FileNotFoundError:
                            pass
                    if not found:
                        print('No batch log yet.')
        else:
            script = (ROOT / args.script).resolve()
            if not script.is_relative_to(ROOT / 'scripts') or script.suffix != '.slurm':
                raise ValueError('Only repository Slurm scripts can be submitted.')
            with client.open_sftp() as sftp:
                home = sftp.normalize('.')
                root = posixpath.join(home, NAME)
            if args.dependency and args.afterany:
                raise ValueError('Choose one dependency type.')
            dependency = (f' --dependency=afterok:{args.dependency}' if args.dependency else
                          f' --dependency=afterany:{args.afterany}' if args.afterany else '')
            output = command(client, 'cd '+shlex.quote(home)+' && sbatch --parsable'+dependency+' --chdir='+shlex.quote(root)
                             +' --export=ALL,HYBRID_REPO_ROOT='+shlex.quote(root)+' '
                             +shlex.quote(posixpath.join(root, script.relative_to(ROOT).as_posix())))
            job = output.strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Submission did not return a numeric Slurm job ID.')
            destination = ROOT / ('docs/hybrid_qualitative_job.json' if 'qualitative' in script.name else
                                  'docs/faculty_setup_job.json' if 'setup' in script.name else
                                  'docs/faculty_continuation_job.json' if 'resume' in script.name else
                                  'docs/faculty_submitted_job.json')
            destination.write_text(json.dumps({'job_id': job, 'script': args.script, 'username': USER}, indent=2)+'\n')


if __name__ == '__main__':
    main()
