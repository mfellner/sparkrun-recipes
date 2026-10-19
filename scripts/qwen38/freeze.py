#!/usr/bin/env python3
"""Freeze raw evidence and harness hashes after capture; approved Git tree anchors it."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    evidence = args.directory.resolve()
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'artifacts': {str(p.relative_to(evidence)): sha(p) for p in sorted(evidence.rglob('*'))
                             if p.is_file() and p != evidence / 'manifest.json' and p != evidence / 'verdict.json'},
                'harness': {str(p.relative_to(root)): sha(p) for p in sorted((root / 'scripts/qwen38').glob('*.py'))}}
    (evidence / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(len(manifest['artifacts']), 'artifacts;', len(manifest['harness']), 'harness scripts')


if __name__ == '__main__':
    main()
