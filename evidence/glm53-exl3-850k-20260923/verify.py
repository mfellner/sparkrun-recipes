#!/usr/bin/env python3
"""Offline, fail-closed semantic verification; all pins external to captured receipts."""
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from contracts import (HOSTS, MODEL, common_parser, launch_argv, load_config, require,
                       requests_for_run, sha, verify_response, verify_http_receipt)
from capture_runtime import commands

REQUIRED_FILES = {'launch.json', 'launch.log', 'before.json', 'acceptance.json', 'http.jsonl', 'after.json'}
VIDEO_DIAGNOSTIC = 'glm53: video placeholders aligned to encoder grid_t (Glm5Next→glm46v)\n'
FATAL = re.compile(r'NVRM:\s*Xid|NV_ERR_NO_MEMORY|out of memory|oom[-_ ]kill|killed process|illegal memory access|CUDA error|NCCL error|unhandled system error|segmentation fault|Traceback \(most recent call last\)', re.I)


def verify_interval(start, end, lower, upper):
    require(all(type(v) in (int, float) and math.isfinite(v) for v in (start, end, lower, upper))
            and lower <= start <= end <= upper, 'invalid timestamp interval')


def verify_kernel(text, since=0, until=None):
    require(bool(text.strip()) and not FATAL.search(text), 'kernel failure/empty capture')
    require(not re.search(r'permission denied|not seeing messages|rotated|no journal files|suppressed|truncat|'
                         r'\b(?:lost|missed|dropped)\b|rate[-_ ]limit', text, re.I),
            'incomplete kernel journal')
    if text == '-- No entries --\n':
        return
    previous = since
    for line in text.splitlines(keepends=True):
        match = re.fullmatch(r'([0-9]+\.[0-9]{6}) [^\s]+ kernel: [^\r\n]+\n', line)
        require(match is not None, 'kernel short-unix structure/truncation')
        stamp = float(match[1])
        require(math.isfinite(stamp) and previous <= stamp and (until is None or stamp <= until),
                'kernel timestamp interval/order')
        previous = stamp


def verify_identity(c, rank, p):
    pins = c['pins']
    require(p['rank'] == rank and p['container'] == c['launch_id'] + '_node_' + str(rank), 'rank/container')
    require(re.fullmatch('[0-9a-f]{64}', p['container_id']), 'container ID')
    require(re.fullmatch('[0-9a-f-]{36}', p['boot_id']), 'boot identity')
    require(p['image'] == pins['image'] and p['image_id'] == p['image_inspect_id'] == pins['image_id']
            and pins['image'] in p['repo_digests'], 'image identity')
    labels = p['labels']
    for key, value in {'sparkrun.cluster_id': c['launch_id'], 'sparkrun.rank': str(rank),
                       'sparkrun.model': c['model'], 'sparkrun.served_model_name': MODEL}.items():
        require(labels.get(key) == value, 'workload label: ' + key)
    state = p['state']
    require(state['Running'] is True and all(state[key] is False for key in ('Paused', 'Restarting', 'OOMKilled', 'Dead')), 'container state')
    require(p['user'] == 'root' and p['security'] == {'Privileged': False, 'NetworkMode': 'host', 'IpcMode': 'host',
            'CapAdd': ['CAP_IPC_LOCK'], 'SecurityOpt': ['no-new-privileges', 'label=disable']}, 'security contract')
    require(p['nccl_version'] == pins['nccl_version'], 'NCCL package version')
    require(p['instanttensor_version'] == pins['instanttensor_version'], 'InstantTensor installed version')
    lines = p['docker_top'].splitlines()
    require(re.fullmatch(r'PID\s+PPID\s+PGID\s+SID\s+COMMAND', lines[0]), 'process table header')
    top = {}
    for line in lines[1:]:
        fields = line.split(None, 4)
        require(len(fields) == 5 and all(x.isdigit() for x in fields[:4]), 'process table row')
        pid = int(fields[0])
        require(pid > 0 and pid not in top, 'duplicate host process')
        top[pid] = {'ppid': int(fields[1]), 'pgid': int(fields[2]), 'sid': int(fields[3]), 'command': fields[4]}
    namespace = {item['pid']: item for item in p['namespace']}
    require(len(namespace) == len(p['namespace']) and {item['host_pid'] for item in namespace.values()} == set(top), 'namespace inventory')
    serving = [row for row in p['processes'] if row['argv'] == c['rank_argv'][str(rank)]]
    workers = [row for row in p['processes'] if ' '.join(row['argv']) == f'VLLM::Worker_TP{rank}']
    require(len(serving) == len(workers) == 1 and len(p['processes']) == 2, 'serving/worker process identity')
    require(all(row['pid'] in namespace for row in p['processes']), 'namespace PID join')
    serve_host = namespace[serving[0]['pid']]['host_pid']
    worker_host = namespace[workers[0]['pid']]['host_pid']
    engines = [pid for pid, row in top.items() if row['command'] == 'VLLM::EngineCore']
    require(len(engines) == (1 if rank == 0 else 0), 'EngineCore lineage inventory')
    require([pid for pid, row in top.items() if row['command'].startswith('VLLM::Worker_TP')] == [worker_host],
            'worker lineage inventory')
    lineage = [serve_host, *engines, worker_host]
    for pid in lineage:
        require(top[pid]['pgid'] == top[pid]['sid'] == serve_host, 'serving lineage group/session')
    for parent, child in zip(lineage, lineage[1:]):
        require(top[child]['ppid'] == parent, 'serving/EngineCore/worker lineage')
    stable = []
    for row in p['processes']:
        ns = namespace[row['pid']]
        require(ns['nspid'] == [ns['host_pid'], row['pid']], 'namespace PID join')
        require(top[ns['host_pid']]['command'] == ' '.join(row['argv']) and ns['start_ticks'] == row['start_ticks'] > 0, 'process table/start join')
        require(row['env'] == c['env'], 'effective process environment')
        for lib in row['nccl']:
            require(lib == {'path': pins['nccl_path'], 'sha256': pins['nccl_sha256']}, 'loaded NCCL identity')
        stable.append((row['pid'], ns['host_pid'], row['start_ticks'], row['argv']))
    require(len(workers[0]['nccl']) == 1, 'worker must map the pinned NCCL library')
    if rank == 0:
        require(len(p['listeners']) == 1 and p['listeners'][0]['owners'] == [serving[0]['pid']], 'direct listener owner')
    else:
        require(p['listeners'] == [], 'headless rank must not listen on API port')
    lineage_state = []
    for pid in lineage:
        ns = next(item for item in namespace.values() if item['host_pid'] == pid)
        require(ns['nspid'] == [pid, ns['pid']] and ns['start_ticks'] > 0, 'lineage namespace/start')
        lineage_state.append({'namespace': ns, 'process': top[pid]})
    return {'container_id': p['container_id'], 'boot_id': p['boot_id'], 'started_at': state['StartedAt'],
            'processes': sorted(stable), 'lineage': lineage_state}


def verify_snapshot(c, snapshot, phase, launch, config_sha):
    require(snapshot['schema'] == 1 and snapshot['phase'] == phase, 'snapshot schema/phase')
    require(snapshot['launch_id'] == c['launch_id'] and snapshot['run_id'] == c['run_id']
            and snapshot['config_sha256'] == config_sha, 'snapshot binding')
    verify_interval(snapshot['started_at'], snapshot['completed_at'], launch['completed_at'], snapshot['completed_at'])
    verify_interval(snapshot['readiness_at'], snapshot['window_until'], launch['started_at'], snapshot['completed_at'])
    require(set(snapshot['hosts']) == set(HOSTS), 'host inventory')
    identities = {}
    for rank, host in enumerate(HOSTS):
        rows = snapshot['hosts'][host]
        expected = commands(c, rank, snapshot['readiness_at'], snapshot['window_until'])
        require(set(rows) == set(expected), 'required host captures')
        for key, argv in expected.items():
            row = rows[key]
            require(row['argv'] == argv, host + ': command binding: ' + key)
            allowed_stderr = ('', VIDEO_DIAGNOSTIC) if key == 'patch_state' else ('',)
            require(row['returncode'] == 0 and row['stderr'] in allowed_stderr, host + ': capture failure: ' + key)
            verify_interval(row['started_at'], row['completed_at'], snapshot['started_at'], snapshot['completed_at'])
        ready = datetime.fromisoformat(rows['ready']['stdout'].strip()).timestamp()
        require(ready == snapshot['readiness_at'] <= snapshot['started_at'], 'readiness receipt')
        payload = json.loads(rows['identity']['stdout'])
        identities[host] = verify_identity(c, rank, payload)
        require(rows['kernel_boot_before']['stdout'] == rows['kernel_boot_after']['stdout'] == payload['boot_id'] + '\n',
                'kernel boot identity join')
        require(snapshot['window_until'] <= rows['kernel']['started_at']
                and rows['kernel_boot_before']['completed_at'] <= rows['kernel']['started_at']
                and rows['kernel']['completed_at'] <= rows['kernel_boot_after']['started_at'],
                'kernel capture interval/boot bracket')
        started = datetime.fromisoformat(payload['state']['StartedAt'].replace('Z', '+00:00')).timestamp()
        require(launch['started_at'] <= started <= ready, 'container launch window')
        manifest = json.loads(rows['mod_manifest']['stdout'])
        require(manifest == {'manifest_sha256': c['pins']['mod_manifest_sha256'], 'files': c['mod_files']}, 'runtime mod identity')
        require(rows['patch_state']['stdout'].strip() == '[OK] complete GLM-5.3 runtime patched state verified', 'installed patch gate')
        log = rows['serve_log']['stdout']
        require(not FATAL.search(log), 'fatal serving log')
        for marker in ('Using fp8_ds_mla KV cache format', 'Graph capturing finished'):
            require(marker in log, 'missing runtime log marker: ' + marker)
        executions = [line.split('[glm53-e3-executed]', 1)[1] for line in log.splitlines()
                      if '[glm53-e3-executed]' in line]
        require(bool(executions) and all(re.fullmatch(
            r' grouped_calls=[1-9][0-9]* fat_expert_runs=[1-9][0-9]* configured_tier=grouped effective_tier=grouped',
            text) for text in executions), 'E3 positive execution counters/grouped tiers required')
        verify_kernel(rows['kernel']['stdout'], snapshot['readiness_at'], snapshot['window_until'])
    return identities


def verify_acceptance(c, a, config_sha, before, after):
    require(a['schema'] == 1 and a['config_sha256'] == config_sha and a['launch_id'] == c['launch_id']
            and a['run_id'] == c['run_id'], 'acceptance binding')
    verify_interval(a['started_at'], a['completed_at'], before['completed_at'], after['started_at'])
    bodies = requests_for_run(c['run_id'])
    require(set(a['checks']) == set(bodies) | {'direct_models', 'proxy_models'}, 'required acceptance matrix')
    for name, row in a['checks'].items():
        verify_http_receipt(name, row)
        verify_interval(row['started_at'], row['completed_at'], a['started_at'], a['completed_at'])
        route = c['proxy_url' if name.startswith('proxy') else 'direct_url']
        url = route + ('/v1/models' if name.endswith('_models') else '/v1/chat/completions')
        require(row['requested_url'] == row['effective_url'] == url, name + ': route/redirect')
        require(json.loads(row['response_text']) == row['response'], name + ': exact response text')
        if name.endswith('_models'):
            response = row['response']
            require(row['http'] == 200 and 'error' not in response and response['object'] == 'list'
                    and sum(m['id'] == MODEL for m in response['data']) == 1, name + ': model listing')
        else:
            require(row['request_text'] == json.dumps(bodies[name]), name + ': exact wire request')
            verify_response(name, row, bodies[name])
    c4 = [a['checks'][f'direct_c4_{i}'] for i in range(4)]
    require(max(row['started_at'] for row in c4) < min(row['completed_at'] for row in c4), 'C4 requests did not overlap')
    require(a['checks']['direct_apc_cold']['completed_at'] <= a['checks']['direct_apc_hot']['started_at'], 'APC replay order')


def verify_bundle(c, root, config_sha):
    require(all((root / name).is_file() for name in REQUIRED_FILES), 'required artifact missing')
    def read(name):
        return json.loads((root / name).read_text())
    launch, before, acceptance, after = [read(name + '.json') for name in ('launch', 'before', 'acceptance', 'after')]
    require(launch['schema'] == 1 and launch['argv'] == launch_argv(c) and launch['returncode'] == 0, 'launch command/status')
    require(launch['pins'] == c['pins'] and launch['config_sha256'] == config_sha
            and launch['launch_id'] == c['launch_id'] and launch['run_id'] == c['run_id'], 'launch pins/identity')
    verify_interval(launch['started_at'], launch['completed_at'], 0, before['started_at'])
    require(launch['launch_log_sha256'] == sha(root / 'launch.log') and (root / 'launch.log').stat().st_size > 0, 'launch log binding')
    require(before['launch_sha256'] == after['launch_sha256'] == sha(root / 'launch.json'), 'launch receipt binding')
    require(acceptance['before_sha256'] == sha(root / 'before.json')
            and after['acceptance_sha256'] == sha(root / 'acceptance.json'), 'capture/acceptance binding')
    before_ids = verify_snapshot(c, before, 'before', launch, config_sha)
    after_ids = verify_snapshot(c, after, 'after', launch, config_sha)
    require(before_ids == after_ids, 'serving process restarted during acceptance')
    require(before['readiness_at'] == after['readiness_at'] and after['window_until'] >= acceptance['completed_at'], 'kernel acceptance coverage')
    verify_acceptance(c, acceptance, config_sha, before, after)
    journal = [json.loads(line) for line in (root / 'http.jsonl').read_text().splitlines()]
    require(len(journal) == len(acceptance['checks']) and len({row['name'] for row in journal}) == len(journal), 'HTTP journal inventory')
    require({row['name']: row['receipt'] for row in journal} == acceptance['checks'], 'HTTP journal exact receipts')
    return {name: sha(root / name) for name in sorted(REQUIRED_FILES)}


def main():
    p = common_parser(__doc__)
    p.add_argument('--write-manifest', action='store_true', help='seal only after full semantic verification')
    a = p.parse_args()
    try:
        c = load_config(a.config, a.expected_config_sha256)
        hashes = verify_bundle(c, a.out_dir, a.expected_config_sha256)
        manifest = ''.join(digest + '  ' + name + '\n' for name, digest in hashes.items())
        path = a.out_dir / 'SHA256SUMS'
        if path.exists():
            require(path.read_text() == manifest, 'receipt manifest mismatch')
        elif a.write_manifest:
            with path.open('x') as stream:
                stream.write(manifest)
        else:
            raise ValueError('receipt manifest missing; use --write-manifest after review')
        print(json.dumps({'passed': True, 'artifacts': len(hashes), 'config_sha256': a.expected_config_sha256,
                          'manifest_sha256': sha(path), 'scope': 'functional, not performance or 850K capacity'}))
        return 0
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        print(json.dumps({'passed': False, 'failures': [str(exc)]}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
