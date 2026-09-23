#!/usr/bin/env python3
"""Offline preparation. Pins must come from the independently reviewed recipe/image."""
import argparse
from pathlib import Path
from contracts import make_config, sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--mod-dir', type=Path, required=True)
    for key in ('recipe-sha256', 'mod-manifest-sha256', 'source-revision', 'image', 'image-id',
                'nccl-sha256', 'nccl-path', 'nccl-version', 'instanttensor-version'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--launch-id', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--cluster', required=True)
    p.add_argument('--direct-url', default='http://127.0.0.1:8000')
    p.add_argument('--proxy-url', default='http://127.0.0.1:4000')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    pins = {key: getattr(a, key) for key in ('recipe_sha256', 'mod_manifest_sha256', 'source_revision',
            'image', 'image_id', 'nccl_sha256', 'nccl_path', 'nccl_version', 'instanttensor_version')}
    config = make_config(a.recipe, a.mod_dir, pins, a.launch_id, a.run_id, a.cluster, a.direct_url, a.proxy_url)
    write_json(a.out, config)
    print(sha(a.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
