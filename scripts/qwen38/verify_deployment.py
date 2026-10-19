#!/usr/bin/env python3
"""Recompute final deployment claims from canonical inputs and raw local receipts.

Run with a Python containing PyYAML (the installed SparkRun environment suffices).
Preliminary loopback receipts and the failed distribution attempt are historical,
not accepted as evidence for the final LAN-serving process.
"""
import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accept import CASES, MODEL, request_for
from verify import REQUIRED, negative_controls, verify

ROOT = Path(__file__).resolve().parents[2]
RECIPE = 'recipes/qwen3.8-27b-tensorfold-dgx-spark.yaml'
RECIPE_SHA = '1e9aa10534862715249e75dff43d21fc9ebf45b615449f241902676caddbe628'
IMAGE = 'ghcr.io/miaai-lab/qwen3.8-27b-dgx-spark-tensorfold@sha256:043d192b9e1725f55c5439fc9672055de891433e5b31448ed8963918e3c460e9'
IMAGE_ID = 'sha256:034aaf3d1c644a2632e959d21a023caba048af0b01e3af7d86d3a63dfa32243f'
CID = '0cbbce67f84eda06c1abe636cb24bc3bce6f8fee73c8d90dad4bf905ad9d7f6e'
WORKLOAD = 'sparkrun_143aaa9f3e323aa9_bcc74ffc19b7'
HOST = '192.168.178.47'
RUNTIME = ['runtime-before.json', 'runtime-after.json', 'runtime-final.json']
PROXY = {n + '.json' for n in ['models', 'health', 'chat', 'vision', 'tools']}
REQUIRED_ROOT = {'launch.json', 'ready.json', 'logs.json', *RUNTIME,
                 'Vontra--Qwen3.8-27B-MLX-4bit.json', 'z-lab--Qwen3.8-27B-DFlash2.json'}


def load(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def command_ok(receipt):
    assert receipt['returncode'] == 0
    assert receipt['started_at'] <= receipt['finished_at']


def check(root):
    manifest = load(root / 'manifest.json')
    required = REQUIRED_ROOT | {'direct/' + n for n in REQUIRED | {'manifest.json'}} | {'proxy/' + n for n in PROXY | {'manifest.json'}}
    assert required <= set(manifest['artifacts']), 'required evidence missing from manifest'
    for rel, sha in manifest['artifacts'].items():
        assert digest(root / rel) == sha, rel
    expected_scripts = {str(p.relative_to(ROOT)) for p in (ROOT / 'scripts/qwen38').glob('*.py')}
    assert set(manifest['harness']) == expected_scripts
    for rel, sha in manifest['harness'].items():
        assert digest(ROOT / rel) == sha, rel
    assert digest(ROOT / RECIPE) == RECIPE_SHA
    recipe = yaml.safe_load((ROOT / RECIPE).read_text())
    launch = load(root / 'launch.json')
    assert launch['returncode'] == 0 and launch['recipe_sha256'] == RECIPE_SHA
    assert launch['earlyoom_before'] == 'inactive' and launch['meminfo_before']['MemAvailable'] >= 109 * 1024**2
    assert launch['argv'] == ['sparkrun', 'run', '/home/max/sparkrun-recipes/' + RECIPE,
        '--hosts', HOST, '-o', 'host=' + HOST, '--trust', '--no-follow', '--no-sync-tuning']
    for rel, sha in launch['input_sha256'].items():
        assert digest(ROOT / rel) == sha, rel
    assert set(launch['input_sha256']) == {RECIPE, *[str(p.relative_to(ROOT)) for p in (ROOT / 'mods/qwen38-tensorfold').iterdir() if p.is_file()]}
    result = verify(root / 'direct')
    assert result['long_prompt_tokens'] == 41443
    assert result['endpoint'] == HOST + ':8000'
    direct = {n: load(root / 'direct' / n) for n in REQUIRED if n.endswith('.json')}
    assert all('raw_response_base64' in r for r in direct.values())
    first = min(r['started_at'] for r in direct.values())
    last = max(r['finished_at'] for r in direct.values())
    assert launch['started_at'] < launch['finished_at'] < first
    order = ['models-before', 'health-before', 'chat']
    for a, b in zip(order, order[1:]):
        assert direct[a + '.json']['finished_at'] <= direct[b + '.json']['started_at']
    assert direct['chat.json']['finished_at'] <= min(direct[c + '.json']['started_at'] for c in CASES[1:9])
    assert max(direct[c + '.json']['finished_at'] for c in CASES[1:9]) <= direct['vision.json']['started_at']
    assert direct['vision.json']['finished_at'] <= direct['tools.json']['started_at']
    assert direct['tools.json']['finished_at'] <= direct['long.json']['started_at']
    assert direct['long.json']['finished_at'] <= direct['health-after.json']['started_at']
    assert direct['health-after.json']['finished_at'] <= direct['models-after.json']['started_at']
    records = []
    for name in RUNTIME:
        r = load(root / name)
        assert r['hostname'] == 'dgx01'
        for key in ['inspect', 'probe', 'top']:
            command_ok(r[key])
        assert r['inspect']['argv'] == ['docker', 'inspect', WORKLOAD + '_solo']
        d = json.loads(r['inspect']['stdout'])[0]
        p = json.loads(r['probe']['stdout'])
        assert d['Id'] == CID and d['Image'] == IMAGE_ID and d['Config']['Image'] == IMAGE
        assert d['State']['Running'] is True and d['State']['OOMKilled'] is False
        assert d['HostConfig']['Privileged'] is False and d['Config']['User'] == 'root'
        assert d['HostConfig']['NetworkMode'] == d['HostConfig']['IpcMode'] == 'host'
        assert d['HostConfig']['Devices'] == [] and d['HostConfig']['CapAdd'] == ['CAP_IPC_LOCK']
        assert 'no-new-privileges' in d['HostConfig']['SecurityOpt'] and not d['Config'].get('Healthcheck')
        assert d['Config']['Labels']['tf.patches'] == 'fb91a0019fdc'
        assert datetime.fromisoformat(d['State']['StartedAt'].replace('Z', '+00:00')).timestamp() < first
        assert len(p['listeners']) == 1
        listener = p['listeners'][0]
        rendered = shlex.split(recipe['command'].format(**{**recipe['defaults'], 'host': HOST}))[3:]
        rendered[rendered.index('--served-model-name')] = '--name'
        assert listener['argv'] == ['/usr/bin/python3', '/usr/local/bin/tensorfold', 'serve', *rendered]
        assert listener['env'] == recipe['env']
        assert p['mods'] == {Path(rel).name: sha for rel, sha in launch['input_sha256'].items() if rel.startswith('mods/')}
        assert p['versions'] == {'tensorfold': '0.6.0', 'torch': '2.13.0a0+9186a08b2c.nv26.7.59513937', 'transformers': '5.17.0', 'av': '19.0.0'}
        assert r['host_pid_binding']['namespace_pids'][-1] == listener['pid']
        assert r['host_pid_binding']['namespace_pids'][0] == r['host_pid_binding']['host_pid']
        assert r['host_pid_binding']['start_ticks'] == listener['start_ticks']
        assert r['earlyoom']['stdout'].strip() == 'inactive'
        assert hashlib.sha256(r['job']['raw'].encode()).hexdigest() == r['job']['sha256']
        job = yaml.safe_load(r['job']['raw'])
        assert job['recipe_state']['_raw'] == recipe
        assert job['cluster_id'] == WORKLOAD and job['hosts'] == [HOST] and job['overrides'] == {'host': HOST}
        assert job['effective_container_image'] == IMAGE and r['job']['mtime'] < first
        records.append((r, d, listener))
    assert records[0][0]['finished_at'] < first
    assert records[1][0]['started_at'] > last
    for r, d, listener in records[1:]:
        assert listener == records[0][2]
        assert r['host_pid_binding'] == records[0][0]['host_pid_binding']
        for key in ['Config', 'HostConfig']:
            assert d[key] == records[0][1][key]
        assert sorted(d['Mounts'], key=lambda m: json.dumps(m, sort_keys=True)) == sorted(records[0][1]['Mounts'], key=lambda m: json.dumps(m, sort_keys=True))
    proxy_manifest = load(root / 'proxy/manifest.json')
    assert set(proxy_manifest) == PROXY
    proxy = {}
    for name in PROXY:
        assert digest(root / 'proxy' / name) == proxy_manifest[name]
        r = load(root / 'proxy' / name)
        assert r['status'] == 200 and 'error' not in r['response']
        assert json.loads(base64.b64decode(r['raw_response_base64'], validate=True)) == r['response']
        assert r['started_at'] > last and r['finished_at'] < records[-1][0]['started_at']
        case = name[:-5]
        assert r['case'] == case
        assert r['url'] == 'http://127.0.0.1:4000' + ('/health' if case == 'health' else '/v1/models' if case == 'models' else '/v1/chat/completions')
        assert r['request'] == (request_for(case) if case in ['chat', 'vision', 'tools'] else None)
        proxy[case] = r
    health = proxy['health']['response']
    assert health['healthy_count'] == 1 and health['unhealthy_count'] == 0 and health['unhealthy_endpoints'] == []
    assert health['healthy_endpoints'][0]['api_base'] == 'http://' + HOST + ':8000/v1'
    assert health['healthy_endpoints'][0]['model'] == 'openai/' + MODEL
    for case in ['chat', 'vision', 'tools']:
        body = proxy[case]['response']
        assert body['model'] == MODEL and body['object'] == 'chat.completion' and body['usage']['completion_tokens'] > 0
        msg = body['choices'][0]['message']
        if case == 'chat': assert msg['content'] == 'SPARK_READY'
        if case == 'vision': assert msg['content'].strip() == 'Left: Red  \nRight: Blue'
        if case == 'tools':
            assert len(msg['tool_calls']) == 1
            fn = msg['tool_calls'][0]['function']
            assert fn['name'] == 'get_weather' and json.loads(fn['arguments']) == {'city': 'Berlin', 'unit': 'celsius'}
    logs = load(root / 'logs.json')
    for key in ['serve', 'serve_sha256', 'kernel_first', 'kernel_second']: command_ok(logs[key])
    assert hashlib.sha256(logs['serve']['stdout'].encode()).hexdigest() == logs['serve_sha256']['stdout'].split()[0]
    serve = logs['serve']['stdout']
    for marker in ['loading Qwen3.8-27B', 'pinned 78.0 GiB cache pool (2,555,904 tokens)', 'serving Qwen3.8-27B at http://' + HOST + ':8000/v1', 'drafts: on', 'prompts: FP8 activations']:
        assert marker in serve, marker
    kernel = logs['kernel_first']['stdout']
    assert kernel == logs['kernel_second']['stdout'] and '-- cursor: ' in kernel
    assert logs['kernel_first']['stderr'] == logs['kernel_second']['stderr'] == ''
    assert logs['since'] <= launch['started_at'] and logs['until'] >= records[-1][0]['finished_at']
    expected_query = ['journalctl', '-k', '--since', '@' + str(logs['since']), '--until', '@' + str(logs['until']), '--no-pager', '-o', 'short-unix', '--show-cursor']
    assert logs['kernel_first']['argv'] == logs['kernel_second']['argv'] == expected_query
    assert re.search(r'^\d+\.\d+ dgx01 kernel:', kernel, re.M)
    assert not re.search(r'Traceback|CUDA error|out of memory|illegal memory|NCCL (?:ERROR|WARN)', serve, re.I)
    assert not re.search(r'NVRM: Xid|Out of memory:|Killed process|oom-kill|NV_ERR_NO_MEMORY', kernel, re.I)
    result.update({'workload': WORKLOAD, 'container': CID, 'configured_context': 262144,
                   'kv_pool_tokens': 2555904, 'proxy_health': '1 healthy, 0 unhealthy', 'runtime_bindings': 'PASS'})
    return result


def deployment_negative_controls(root):
    import shutil
    import tempfile
    def rejected(change):
        with tempfile.TemporaryDirectory(prefix='qwen38-deployment-negative-', dir=root.parent) as tmp:
            p = Path(tmp) / 'evidence'
            shutil.copytree(root, p)
            change(p)
            manifest = load(p / 'manifest.json')
            for rel in list(manifest['artifacts']):
                if (p / rel).exists():
                    manifest['artifacts'][rel] = digest(p / rel)
                else:
                    del manifest['artifacts'][rel]
            (p / 'manifest.json').write_text(json.dumps(manifest))
            try:
                check(p)
            except (AssertionError, KeyError, ValueError, FileNotFoundError):
                return
            raise AssertionError('deployment mutation unexpectedly passed')
    def runtime(p, field):
        file = p / 'runtime-before.json'
        r = load(file)
        if field == 'security':
            d = json.loads(r['inspect']['stdout'])
            d[0]['HostConfig']['Privileged'] = True
            r['inspect']['stdout'] = json.dumps(d)
        elif field == 'pid':
            r['host_pid_binding']['start_ticks'] += 1
        elif field == 'job':
            job = yaml.safe_load(r['job']['raw'])
            job['overrides']['host'] = '127.0.0.1'
            r['job']['raw'] = yaml.safe_dump(job)
            r['job']['sha256'] = hashlib.sha256(r['job']['raw'].encode()).hexdigest()
        file.write_text(json.dumps(r))
    for field in ['security', 'pid', 'job']:
        rejected(lambda p, field=field: runtime(p, field))
    def missing(p):
        (p / 'runtime-final.json').unlink()
    rejected(missing)
    def kernel(p):
        file = p / 'logs.json'
        r = load(file)
        for key in ['kernel_first', 'kernel_second']:
            r[key]['stdout'] = '1792370100.000000 dgx01 kernel: NV_ERR_NO_MEMORY\n' + r[key]['stdout']
        file.write_text(json.dumps(r))
    rejected(kernel)
    def timestamp(p):
        file = p / 'runtime-before.json'
        r = load(file)
        r['finished_at'] += 1000000
        file.write_text(json.dumps(r))
    rejected(timestamp)
    return 6


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--negative-controls', action='store_true')
    args = parser.parse_args()
    result = check(args.directory)
    if args.negative_controls:
        result['semantic_negative_controls_rejected'] = negative_controls(args.directory / 'direct')
        result['deployment_negative_controls_rejected'] = deployment_negative_controls(args.directory)
    print(json.dumps(result, indent=2))
