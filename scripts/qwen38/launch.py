#!/usr/bin/env python3
"""Guard an idle local Spark launch and preserve exact inputs and command output."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / 'recipes/qwen3.8-27b-tensorfold-dgx-spark.yaml'
MOD = ROOT / 'mods/qwen38-tensorfold'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='192.168.178.47', help='SparkRun target; this script must run on that host')
    parser.add_argument('--bind', default='127.0.0.1', help='Inference bind address; use a trusted private LAN IPv4 for proxy discovery')
    args = parser.parse_args()
    import ipaddress
    bind = ipaddress.ip_address(args.bind)
    assert bind.version == 4 and (bind.is_loopback or bind.is_private) and not bind.is_unspecified
    assert subprocess.run(['systemctl', 'is-active', 'earlyoom'], capture_output=True, text=True).stdout.strip() == 'inactive'
    mem = {l.split(':')[0]: int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.split()[1].isdigit()}
    assert mem['MemAvailable'] >= 109 * 1024**2, 'require 109 GiB MemAvailable'
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 8000)) != 0, 'port 8000 already occupied'
    files = [RECIPE, *sorted(p for p in MOD.iterdir() if p.is_file())]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    argv = ['sparkrun', 'run', str(RECIPE), '--hosts', args.host, '-o', 'host=' + args.bind, '--trust', '--no-follow', '--no-sync-tuning']
    if args.output.exists():
        raise SystemExit('refusing to overwrite launch receipt')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    receipt = {'argv': argv, 'started_at': started, 'finished_at': time.time(), 'returncode': result.returncode,
               'stdout': result.stdout, 'stderr': result.stderr, 'input_sha256': hashes,
               'recipe_sha256': hashes[str(RECIPE.relative_to(ROOT))], 'meminfo_before': mem, 'earlyoom_before': 'inactive'}
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(result.stdout, result.stderr, flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
