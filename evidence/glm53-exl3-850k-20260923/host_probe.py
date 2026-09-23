#!/usr/bin/env python3
"""Read-only host/container probe, transported verbatim over SSH by the collector."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CONTAINER_PROBE = r'''
import hashlib, importlib.metadata, json, os, pathlib, sys, sysconfig
# Read installed metadata explicitly without running site/.pth startup hooks.
sys.path.extend(sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')}))
keys=json.loads(sys.argv[1]); rows=[]
for directory in pathlib.Path('/proc').iterdir():
    if not directory.name.isdigit(): continue
    try:
        argv=[x.decode() for x in (directory/'cmdline').read_bytes().split(b'\0') if x]
        title=' '.join(argv)
        if not ('/usr/local/bin/vllm' in argv or title.startswith('VLLM::Worker_TP')): continue
        stat=(directory/'stat').read_text().rsplit(') ',1)[1].split()
        env=dict(x.decode().split('=',1) for x in (directory/'environ').read_bytes().split(b'\0') if b'=' in x)
        libraries=sorted({line.split()[-1] for line in (directory/'maps').read_text().splitlines() if 'libnccl.so' in line})
        nccl=[]
        for lib in libraries:
            p=pathlib.Path(lib).resolve()
            nccl.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        rows.append({'pid':int(directory.name),'start_ticks':int(stat[19]),'argv':argv,
                     'env':{key:env.get(key) for key in keys}, 'nccl':nccl})
    except FileNotFoundError: continue
listeners=[]
for table in ('/proc/net/tcp','/proc/net/tcp6'):
    for line in pathlib.Path(table).read_text().splitlines()[1:]:
        fields=line.split()
        if fields[3]!='0A' or int(fields[1].split(':')[1],16)!=8000: continue
        inode=fields[9]; owners=[]
        for row in rows:
            try:
                if any(os.readlink(fd)=='socket:['+inode+']' for fd in (pathlib.Path('/proc')/str(row['pid'])/'fd').iterdir()):
                    owners.append(row['pid'])
            except FileNotFoundError: pass
        listeners.append({'inode':inode,'owners':owners})
print(json.dumps({'processes':rows,'listeners':listeners,
                  'nccl_version':importlib.metadata.version('nvidia-nccl-cu13'),
                  'instanttensor_version':importlib.metadata.version('instanttensor')}))
'''


def run(argv):
    result = subprocess.run(argv, text=True, capture_output=True)
    # Forward before checking: the outer command receipt must retain failures.
    sys.stderr.write(result.stderr)
    if result.returncode or result.stderr:
        raise ValueError('nested probe failed: status/stderr: ' + str(result.returncode))
    return result.stdout


def main():
    container, rank, env_json, image = sys.argv[1:]
    inspect = json.loads(run(['docker', 'inspect', container]))[0]
    image_data = json.loads(run(['docker', 'image', 'inspect', image]))[0]
    top = run(['docker', 'top', container, '-eo', 'pid,ppid,pgid,sid,args'])
    namespace = []
    for line in top.splitlines()[1:]:
        host_pid = int(line.split()[0])
        proc = Path('/proc') / str(host_pid)
        status = proc.joinpath('status').read_text()
        nspid = next(x for x in status.splitlines() if x.startswith('NSpid:')).split()[1:]
        stat = proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
        namespace.append({'host_pid': host_pid, 'pid': int(nspid[-1]),
                          'start_ticks': int(stat[19]), 'nspid': [int(x) for x in nspid]})
    runtime = json.loads(run(['docker', 'exec', container, 'python3', '-S', '-c', CONTAINER_PROBE, env_json]))
    # Never serialize the full container environment or unrelated OCI labels.
    payload = {'container': container, 'rank': int(rank), 'container_id': inspect['Id'],
               'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               'image': inspect['Config']['Image'], 'image_id': inspect['Image'],
               'image_inspect_id': image_data['Id'], 'repo_digests': image_data['RepoDigests'],
               'labels': {k: v for k, v in inspect['Config']['Labels'].items() if k.startswith('sparkrun.')},
               'user': inspect['Config']['User'], 'state': inspect['State'],
               'security': {k: inspect['HostConfig'][k] for k in ('Privileged', 'NetworkMode', 'IpcMode', 'CapAdd', 'SecurityOpt')},
               'docker_top': top, 'namespace': namespace, **runtime}
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
