#!/usr/bin/env python3
"""Pinned drafter staging for SparkRun 0.3.6's broken local per-entry revisions."""
import hashlib
import json
import os
from pathlib import Path

REPO = 'z-lab/Qwen3.8-27B-DFlash2'
REVISION = '50307d4c4cde6860d4eee73e2547cd786fe8e8a4'
FILES = {
    'config.json': (1239, '873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980'),
    'model.safetensors': (3848817896, '67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c'),
}


def verify(snapshot):
    for name, (size, digest) in FILES.items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != size:
            return False
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != digest:
            return False
    return True


def main():
    cache = Path(os.environ.get('HF_HOME', '/cache/huggingface')) / 'hub'
    snapshot = cache / 'models--z-lab--Qwen3.8-27B-DFlash2' / 'snapshots' / REVISION
    if not verify(snapshot):
        # No token forwarding; this public repository needs no authentication.
        os.environ['HF_HUB_OFFLINE'] = '0'
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, cache_dir=str(cache),
                          allow_patterns=list(FILES), force_download=True, max_workers=2, token=False)
        if not verify(snapshot):
            raise RuntimeError('pinned drafter integrity check failed')
    print(json.dumps({'drafter': REPO, 'revision': REVISION, 'verified': True}), flush=True)


if __name__ == '__main__':
    main()
