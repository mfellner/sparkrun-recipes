#!/usr/bin/env python3
"""Record reproducible lightweight validation without loading a second model."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    recipe = root / 'recipes/qwen3.8-27b-tensorfold-dgx-spark.yaml'
    commands = [
        ['sparkrun', 'recipe', 'validate', str(recipe), '--json'],
        ['sparkrun', 'run', str(recipe), '--hosts', '192.168.178.47', '-o', 'host=192.168.178.47', '--trust', '--dry-run', '--no-sync-tuning'],
        ['python', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_qwen38_tensorfold_adapter.py', '-v'],
        ['bash', '-n', 'mods/qwen38-tensorfold/run.sh'],
    ]
    records = []
    for argv in commands:
        start = time.time()
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True)
        records.append({'argv': argv, 'started_at': start, 'finished_at': time.time(),
                        'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    args.output.write_text(json.dumps(records, indent=2) + '\n')
    assert all(r['returncode'] == 0 for r in records), records
    print('recipe validation, final-LAN dry run, adapter unit tests and shell syntax: PASS')


if __name__ == '__main__':
    main()
