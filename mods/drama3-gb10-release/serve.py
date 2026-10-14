#!/usr/bin/env python3
"""Minimal compact-head wrapper around the unchanged Fish LAN adapter. MIT."""
import argparse
import hashlib
import json
import os
from pathlib import Path

BOOL_DISABLED = ('FISH_DRAMA3_SMALL_GRAPH_BUCKETS', 'FISH_DRAMA3_PREFIX_SNAPSHOT',
                 'DRAMA_NATIVE_STREAMING', 'DRAMA_PUBLIC_REFERENCE_CACHE', 'DRAMA_STATEFUL_CODEC',
                 'FISH_DRAMA3_KERNEL_STATE', 'FISH_DRAMA3_KERNEL_CONV', 'FISH_DRAMA3_KERNEL_FP8')
QUANT_DISABLED = ('FISH_DRAMA3_KERNEL_TOWER_UP', 'FISH_DRAMA3_KERNEL_GDN_QKV')


def profile_from_env(env):
    head = env.get('FISH_DRAMA3_COMPACT_HEAD', '1')
    if head not in ('0', '1'):
        raise ValueError('FISH_DRAMA3_COMPACT_HEAD must be 0 or 1')
    for name in BOOL_DISABLED:
        if env.get(name, '0') != '0':
            raise ValueError(name + ' is not admitted by the minimal release')
    for name in QUANT_DISABLED:
        if env.get(name, 'off') != 'off':
            raise ValueError(name + ' is not admitted by the minimal release')
    return {'configured_flags': {'FISH_DRAMA3_COMPACT_HEAD': head == '1',
                                **{name: False for name in BOOL_DISABLED[:5]}},
            'kernel_options': {'state': False, 'conv': False, 'fp8': False,
                               'tower_up': 'off', 'gdn_qkv': 'off'},
            'scope': 'effective configuration and physical source identity; not a qualification verdict'}


def main():
    configuration = profile_from_env(os.environ)
    # Default also reaches the actual native loader, not merely the profile endpoint.
    os.environ.setdefault('FISH_DRAMA3_COMPACT_HEAD', '1')
    import server
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--served-model-name', choices=[server.MODEL_ID], default=server.MODEL_ID)
    args = parser.parse_args()
    host, port = server.validate_binding(args.host, args.port)
    root = Path(__file__).resolve().parent
    pins = json.loads((root / 'perf/source-pin.json').read_text())
    files = {**pins['before'], **pins['after']}
    configuration['source_files'] = {name: hashlib.sha256((Path('/opt/fish-speech') / name).read_bytes()).hexdigest() for name in files}
    if configuration['source_files'] != files:
        raise ValueError('release source drift')
    configuration['package_manifest_sha256'] = hashlib.sha256((root / 'manifest.sha256').read_bytes()).hexdigest()

    @server.app.get('/v1/runtime-profile')
    def runtime_profile():
        return configuration

    print(json.dumps({'DRAMA3_CONFIGURED_PROFILE': configuration}), flush=True)
    server.uvicorn.run(server.app, host=host, port=port, workers=1,
                       access_log=False, limit_concurrency=16)


if __name__ == '__main__':
    main()
