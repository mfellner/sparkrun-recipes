#!/usr/bin/env python3
"""Before/after read-only snapshots; exact SSH commands bind rank/log/kernel evidence."""
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from contracts import HOSTS, ROOT, common_parser, load_config, require, sha, write_json

MOD = '/workspace/mods/glm-5.3-flash-exl3-upstream-850k'


def ssh_argv(host, command):
    return ['ssh', '-o', 'BatchMode=yes', host, 'bash', '-lc', shlex.quote(command)]


def commands(c, rank, since, until):
    container = c['launch_id'] + '_node_' + str(rank)
    q = shlex.quote(container)
    identity = shlex.join(['python3', '-S', '-c', (ROOT / 'host_probe.py').read_text(), container,
                          str(rank), json.dumps(sorted(c['env'])), c['pins']['image']])
    manifest_script = ("import hashlib,json,pathlib; p=pathlib.Path(" + repr(MOD) + "); "
        "text=(p/'SHA256SUMS').read_text(); "
        "print(json.dumps({'manifest_sha256':hashlib.sha256((p/'SHA256SUMS').read_bytes()).hexdigest(),"
        "'files':{line.split('  ',1)[1]:hashlib.sha256((p/line.split('  ',1)[1]).read_bytes()).hexdigest() "
        "for line in text.splitlines() if line}}))")
    base = {'identity': identity,
            'kernel_boot_before': 'cat /proc/sys/kernel/random/boot_id',
            'kernel': f"journalctl -k -b 0 --since '@{since:.6f}' --until '@{until:.6f}' --no-pager -o short-unix",
            'kernel_boot_after': 'cat /proc/sys/kernel/random/boot_id',
            'serve_log': f'docker exec {q} cat /tmp/sparkrun_serve.log',
            'patch_state': f'docker exec {q} python3 {MOD}/verify_runtime_patch_state.py',
            'mod_manifest': shlex.join(['docker', 'exec', container, 'python3', '-S', '-c', manifest_script]),
            'ready': f'docker exec {shlex.quote(c["launch_id"] + "_node_0")} cat /tmp/glm53-postready.ok'}
    return {key: ssh_argv(HOSTS[0] if key == 'ready' else HOSTS[rank], value) for key, value in base.items()}


def run(argv):
    start = time.time()
    process = subprocess.run(argv, text=True, capture_output=True, timeout=600)
    return {'argv': argv, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr,
            'started_at': start, 'completed_at': time.time()}


def main():
    p = common_parser(__doc__)
    p.add_argument('--phase', choices=('before', 'after'), required=True)
    a = p.parse_args()
    c = load_config(a.config, a.expected_config_sha256)
    launch_path = a.out_dir / 'launch.json'
    launch = json.loads(launch_path.read_text())
    require(launch['config_sha256'] == a.expected_config_sha256 and launch['returncode'] == 0, 'successful bound launch required')
    target = a.out_dir / (a.phase + '.json')
    require(not target.exists(), 'refuse overwrite')
    ready_argv = commands(c, 0, launch['started_at'], time.time())['ready']
    ready = run(ready_argv)
    require(ready['returncode'] == 0 and not ready['stderr'], 'postready marker not available')
    readiness = datetime.fromisoformat(ready['stdout'].strip()).timestamp()
    require(launch['started_at'] <= readiness <= time.time(), 'readiness outside launch window')
    until = time.time()
    record = {'schema': 1, 'config_sha256': a.expected_config_sha256, 'launch_id': c['launch_id'],
              'run_id': c['run_id'], 'launch_sha256': sha(launch_path), 'phase': a.phase,
              'started_at': ready['started_at'], 'readiness_at': readiness, 'window_until': until, 'hosts': {}}
    if a.phase == 'after':
        acceptance_path = a.out_dir / 'acceptance.json'
        acceptance = json.loads(acceptance_path.read_text())
        require(acceptance['completed_at'] <= until, 'acceptance not completed')
        record['acceptance_sha256'] = sha(acceptance_path)
    try:
        for rank, host in enumerate(HOSTS):
            record['hosts'][host] = {}
            for name, argv in commands(c, rank, readiness, until).items():
                record['hosts'][host][name] = run(argv)
    finally:
        record['completed_at'] = time.time()
        write_json(target, record)
    print(target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
