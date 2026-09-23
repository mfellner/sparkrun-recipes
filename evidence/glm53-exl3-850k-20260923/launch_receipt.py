#!/usr/bin/env python3
"""Launch only when explicitly invoked; preserve real stdout/stderr and timestamps."""
import json
import os
import subprocess
import time
from contracts import common_parser, launch_argv, load_config, require, sha, write_json


def main():
    p = common_parser(__doc__)
    p.add_argument('--execute-launch', action='store_true', help='required acknowledgement of deployment side effect')
    a = p.parse_args()
    c = load_config(a.config, a.expected_config_sha256)
    require(a.execute_launch, 'not launching without --execute-launch')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(a.out_dir, 0o700)
    require(not any(a.out_dir.iterdir()), 'launch requires a new empty receipt directory')
    argv = launch_argv(c)
    record = {'schema': 1, 'config_sha256': a.expected_config_sha256, 'launch_id': c['launch_id'],
              'run_id': c['run_id'], 'argv': argv, 'pins': c['pins'], 'started_at': time.time()}
    # Direct file recording preserves partial output even if the controller dies.
    with (a.out_dir / 'launch.log').open('x') as log:
        process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, text=True)
    record.update(returncode=process.returncode, completed_at=time.time(), launch_log_sha256=sha(a.out_dir / 'launch.log'))
    write_json(a.out_dir / 'launch.json', record)
    print(json.dumps({'returncode': process.returncode, 'launch_id': c['launch_id']}))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
