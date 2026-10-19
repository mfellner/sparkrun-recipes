#!/usr/bin/env python3
"""Capture the exact local container/process/job behind the loopback API."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time

CONTAINER = 'sparkrun_143aaa9f3e323aa9_bcc74ffc19b7_solo'
ROOT = Path(__file__).resolve().parents[2]
PROBE = r'''
import glob, hashlib, importlib.metadata as md, json, os, pathlib
port = 8000
inodes = set()
for name in ['tcp', 'tcp6']:
 for row in pathlib.Path('/proc/net/'+name).read_text().splitlines()[1:]:
  x=row.split()
  if int(x[1].split(':')[1],16)==port and x[3]=='0A': inodes.add(x[9])
listeners=[]
for path in pathlib.Path('/proc').iterdir():
 if not path.name.isdigit(): continue
 try:
  if not any(os.readlink(fd) in {'socket:['+n+']' for n in inodes} for fd in (path/'fd').iterdir()): continue
  stat=(path/'stat').read_text(); start=int(stat.rsplit(')',1)[1].split()[19])
  env={e.split('=',1)[0]:e.split('=',1)[1] for e in (path/'environ').read_bytes().decode().split('\x00') if '=' in e}
  listeners.append({'pid':int(path.name),'start_ticks':start,'argv':(path/'cmdline').read_bytes().decode().split('\x00')[:-1],
    'env':{k:v for k,v in env.items() if k.startswith('TENSORFOLD_') or k in ['HF_HOME','HF_HUB_OFFLINE','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']}})
 except (OSError,ProcessLookupError): continue
mods={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path('/workspace/mods/qwen38-tensorfold').iterdir() if p.is_file()}
import tensorfold
package=pathlib.Path(tensorfold.__file__).parent
sources={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*.py'))}
print(json.dumps({'listeners':listeners,'mods':mods,'versions':{n:md.version(n) for n in ['tensorfold','torch','transformers','av']},
 'package_source_sha256':hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()}))
'''


def command(argv):
    started = time.time()
    p = subprocess.run(argv, capture_output=True, text=True)
    return {'argv': argv, 'started_at': started, 'finished_at': time.time(),
            'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def safe(text):
    # These probes must not publish credentials. Fail before writing, never guess redactions.
    patterns = [r'(?i)(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY|AWS_SECRET_ACCESS_KEY)=\S+',
                r'\bhf_[A-Za-z0-9]{20,}\b', r'gh[pousr]_[A-Za-z0-9]{20,}',
                r'(?i)https?://[^\s/@]+:[^\s/@]+@']
    assert not any(re.search(pattern, text) for pattern in patterns), 'unexpected credential-shaped capture; do not publish'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite runtime receipt')
    started = time.time()
    inspect = command(['docker', 'inspect', CONTAINER])
    assert inspect['returncode'] == 0
    state = json.loads(inspect['stdout'])[0]
    probe = command(['docker', 'exec', CONTAINER, 'python', '-c', PROBE])
    assert probe['returncode'] == 0
    listener = json.loads(probe['stdout'])['listeners']
    assert len(listener) == 1 and 'serve' in listener[0]['argv']
    top = command(['docker', 'top', CONTAINER, '-eo', 'pid,ppid,args'])
    candidates = []
    for row in top['stdout'].splitlines()[1:]:
        hostpid = int(row.split()[0])
        path = Path('/proc') / str(hostpid)
        status = (path / 'status').read_text()
        nspid = [int(i) for i in next(l for l in status.splitlines() if l.startswith('NSpid:')).split()[1:]]
        start_ticks = int((path/'stat').read_text().rsplit(')',1)[1].split()[19])
        if nspid[-1] == listener[0]['pid'] and start_ticks == listener[0]['start_ticks']:
            candidates.append({'host_pid': hostpid, 'namespace_pids': nspid, 'start_ticks': start_ticks})
    assert len(candidates) == 1
    job = Path.home() / '.cache/sparkrun/jobs/143aaa9f3e323aa9_bcc74ffc19b7.yaml'
    receipt = {'started_at': started, 'finished_at': time.time(), 'hostname': socket.gethostname(),
        'inspect': inspect, 'probe': probe, 'top': top, 'host_pid_binding': candidates[0],
        'job': {'path': str(job), 'mtime': job.stat().st_mtime, 'sha256': hashlib.sha256(job.read_bytes()).hexdigest(), 'raw': job.read_text()},
        'earlyoom': command(['systemctl', 'is-active', 'earlyoom']),
        'gpu': command(['nvidia-smi', '--query-gpu=name,pstate,clocks.sm,power.draw,utilization.gpu', '--format=csv']),
        'meminfo': Path('/proc/meminfo').read_text()}
    payload = json.dumps(receipt, indent=2) + '\n'
    safe(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(json.dumps({'container': state['Id'], 'listener': listener[0], 'binding': candidates[0]}, indent=2))


if __name__ == '__main__':
    main()
