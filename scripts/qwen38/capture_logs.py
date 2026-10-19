#!/usr/bin/env python3
"""Capture and independently re-read full serve and bounded kernel logs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

CONTAINER = 'sparkrun_143aaa9f3e323aa9_bcc74ffc19b7_solo'


def run(argv):
    started = time.time()
    result = subprocess.run(argv, capture_output=True)
    return {'argv': argv, 'started_at': started, 'finished_at': time.time(),
            'returncode': result.returncode, 'stdout': result.stdout.decode(), 'stderr': result.stderr.decode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    assert not (root / 'logs.json').exists()
    launch = json.loads((root / 'launch.json').read_text())
    since = int(launch['started_at']) - 60
    until = int(time.time()) + 1
    time.sleep(max(0, until - time.time()))
    first = run(['docker', 'exec', CONTAINER, 'cat', '/tmp/sparkrun_serve.log'])
    second = run(['docker', 'exec', CONTAINER, 'sha256sum', '/tmp/sparkrun_serve.log'])
    assert first['returncode'] == second['returncode'] == 0
    assert hashlib.sha256(first['stdout'].encode()).hexdigest() == second['stdout'].split()[0]
    argv = ['journalctl', '-k', '--since', '@' + str(since), '--until', '@' + str(until),
            '--no-pager', '-o', 'short-unix', '--show-cursor']
    kernel1, kernel2 = run(argv), run(argv)
    assert kernel1['returncode'] == kernel2['returncode'] == 0
    assert kernel1['stdout'] == kernel2['stdout'] and kernel1['stderr'] == kernel2['stderr'] == ''
    assert '-- cursor: ' in kernel1['stdout']
    (root / 'logs.json').write_text(json.dumps({'serve': first, 'serve_sha256': second,
        'kernel_first': kernel1, 'kernel_second': kernel2, 'since': since, 'until': until}, indent=2) + '\n')
    print('Captured identical bounded kernel queries and full serve log with independent hash')


if __name__ == '__main__':
    main()
