"""CPU-only contract/mutation tests. Synthetic fixtures are NOT live evidence."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / 'evidence/glm53-exl3-850k-20260923'


def load_harness():
    assert (HARNESS / 'contracts.py').is_file(), 'refreshed contracts are missing'
    sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location('refresh_contracts', HARNESS / 'contracts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reviewed_defaults_reject_old_scheduler_and_implicit_kv():
    h = load_harness()
    env = {'GLM53_MIXED_PREFILL_CHUNK': 'fair', 'GLM53_DRAFT_KV_COMPACT': '0',
           'DEFAULT_MAX_NEW_TOKENS': '65536', 'SPT_NOENV': '1'}
    argv = ['--load-format', 'instanttensor', '--kv-cache-memory-bytes', '15032385536',
            '--tensor-parallel-size', '2', '--max-model-len', '850000',
            '--max-num-seqs', '4', '--enable-prefix-caching',
            '--allowed-media-domains', 'media.invalid', '--limit-mm-per-prompt', '{"image":4,"video":0}']
    h.validate_defaults(env, argv)
    for key, value in [('GLM53_MIXED_PREFILL_CHUNK', 'skip'), ('GLM53_DRAFT_KV_COMPACT', '1'),
                       ('DEFAULT_MAX_NEW_TOKENS', '0'), ('SPT_NOENV', None), ('SPT_NOENV', '0')]:
        with pytest.raises(ValueError):
            h.validate_defaults({**env, key: value}, argv)
    with pytest.raises(ValueError):
        h.validate_defaults(env, argv[4:])


def test_canonical_requests_preserve_matrix_and_new_controls():
    h = load_harness()
    bodies = h.requests_for_run('0123456789abcdef')
    assert bodies['direct_tool_none']['tool_choice'] == 'none'
    assert bodies['direct_tool_call']['tool_choice'] == 'auto'
    assert 'max_tokens' not in bodies['direct_omitted_limit']
    assert 'max_completion_tokens' not in bodies['direct_omitted_limit']
    assert bodies['direct_long_output']['max_tokens'] > 2000
    assert bodies['direct_long_context']['messages'][0]['content'].count('alpha ') == 110000
    assert bodies['direct_json_reasoning']['response_format']['type'] == 'json_schema'
    assert bodies['direct_apc_cold'] == bodies['direct_apc_hot']
    assert len([key for key in bodies if key.startswith('direct_c4_')]) == 4
    for name in ['direct_exact', 'proxy_exact', 'direct_ocr', 'proxy_ocr',
                 'direct_remote_media_rejected', 'proxy_remote_media_rejected',
                 'direct_video_rejected', 'proxy_video_rejected', 'direct_reasoning_open_stop']:
        assert name in bodies


def test_disabled_thinking_stop_predeclares_ten_constrained_requests():
    h = load_harness()
    bodies = h.requests_for_run('0123456789abcdef')
    names = [name for name in bodies if name.startswith('direct_thinking_disabled_stop')]
    assert names == [f'direct_thinking_disabled_stop_{i}' for i in range(10)]
    expected = h.legacy_acceptance().thinking_disabled_stop_request()
    expected.update(ignore_eos=True, structured_outputs={'regex': 'BEFORE Question: AFTER'})
    for name in names:
        assert bodies[name] == expected
    assert len({id(bodies[name]) for name in names}) == 10


@pytest.mark.parametrize('index', range(10))
def test_every_disabled_thinking_receipt_rejects_semantic_and_request_mutations(tmp_path, index):
    import base64
    import json
    h, v = load_harness(), load_component('verify')
    c, digest = unit_bundle(tmp_path)
    v.verify_bundle(c, tmp_path, digest)
    path = tmp_path / 'acceptance.json'
    original = json.loads(path.read_text())
    name = f'direct_thinking_disabled_stop_{index}'
    mutations = [
        ('literal stop', lambda row: row['response']['choices'][0]['message'].update(content='BEFORE')),
        ('literal stop', lambda row: row['response']['choices'][0]['message'].update(content=' BEFORE ')),
        ('literal stop', lambda row: row['response']['choices'][0]['message'].update(content='BEFORE Question: AFTER')),
        ('literal stop', lambda row: row['response']['choices'][0]['message'].update(content='BEFORE extra prose')),
        ('literal stop', lambda row: row['response']['choices'][0].update(finish_reason='length')),
        ('literal stop', lambda row: row['response']['choices'][0].update(stop_reason=154827)),
        ('literal stop', lambda row: row['response']['choices'][0].update(stop_reason=None)),
        ('literal stop', lambda row: row['response']['choices'][0].pop('stop_reason')),
        ('exact wire request', lambda row: row['request'].pop('ignore_eos')),
        ('exact wire request', lambda row: row['request'].update(ignore_eos=False)),
        ('exact wire request', lambda row: row['request'].pop('structured_outputs')),
        ('exact wire request', lambda row: row['request'].update(structured_outputs={'regex': '.*'})),
        ('exact wire request', lambda row: row['request'].update(stop=[])),
        ('exact wire request', lambda row: row['request'].update(chat_template_kwargs={'enable_thinking': True})),
    ]
    for expected_error, mutate in mutations:
        changed = copy.deepcopy(original)
        row = changed['checks'][name]
        mutate(row)
        row['request_text'] = json.dumps(row['request'])
        row['request_sha256'] = h.request_sha(row['request'])
        row['response_text'] = json.dumps(row['response'])
        row['response_base64'] = base64.b64encode(row['response_text'].encode()).decode('ascii')
        path.write_text(json.dumps(changed))
        refresh_unit_bindings(tmp_path, h)
        with pytest.raises(ValueError, match=expected_error):
            v.verify_bundle(c, tmp_path, digest)
    # No best-of-ten selection or replacement with the obsolete single probe.
    for replacement in (None, 'direct_thinking_disabled_stop'):
        changed = copy.deepcopy(original)
        row = changed['checks'].pop(name)
        if replacement is not None:
            changed['checks'][replacement] = row
        path.write_text(json.dumps(changed))
        refresh_unit_bindings(tmp_path, h)
        with pytest.raises(ValueError, match='required acceptance matrix'):
            v.verify_bundle(c, tmp_path, digest)


def test_historical_exact_receipt_semantics_and_refreshed_hash_mutations():
    h = load_harness()
    previous = __import__('json').loads((ROOT / 'evidence/glm53-exl3-850k-20260913/acceptance.json').read_text())
    row = previous['checks']['direct_exact']
    body = h.requests_for_run('0123456789abcdef')['direct_exact']
    h.verify_response('direct_exact', row, body)
    for mutate in [lambda r: r.update(http=500),
                   lambda r: r['response'].update(model='other'),
                   lambda r: r['response'].update(error={'message': 'failed'}),
                   lambda r: r['response']['choices'][0]['message'].update(content='wrong'),
                   lambda r: r['request'].update(temperature=1)]:
        changed = copy.deepcopy(row)
        mutate(changed)
        changed['request_sha256'] = h.request_sha(changed['request'])
        with pytest.raises(ValueError):
            h.verify_response('direct_exact', changed, body)


@pytest.fixture
def config_inputs(tmp_path):
    import json
    import yaml
    h = load_harness()
    previous = json.loads((ROOT / 'evidence/glm53-exl3-850k-20260913/launch-receipt.json').read_text())
    command = previous['recipe_command'].replace('--quantization exl3', '--load-format instanttensor --kv-cache-memory-bytes 15032385536 --quantization exl3')
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(yaml.safe_dump({'command': command, 'container': 'example/image@sha256:' + 'a'*64,
        'model': 'test/model', 'metadata': {'source_revision': 'b'*40},
        'env': {'GLM53_MIXED_PREFILL_CHUNK': 'fair', 'GLM53_DRAFT_KV_COMPACT': '0', 'DEFAULT_MAX_NEW_TOKENS': '65536', 'SPT_NOENV': '1'}}))
    mod = tmp_path / 'mod'
    mod.mkdir()
    (mod / 'sample.py').write_text('# test-only fixture\n')
    (mod / 'SHA256SUMS').write_text(h.sha(mod / 'sample.py') + '  sample.py\n')
    pins = {'recipe_sha256': h.sha(recipe), 'mod_manifest_sha256': h.sha(mod / 'SHA256SUMS'),
            'source_revision': 'b'*40, 'image': 'example/image@sha256:' + 'a'*64,
            'image_id': 'sha256:' + 'c'*64, 'nccl_sha256': 'd'*64, 'nccl_version': '2.30.7',
            'instanttensor_version': '0.2.0',
            'nccl_path': '/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2'}
    return h, recipe, mod, pins


@pytest.mark.parametrize('launch_id', [
    'sparkrun_0123456789abcde_20260923abcd',  # short intent
    'sparkrun_0123456789abcdef0_20260923abcd',  # long intent
    'sparkrun_0123456789abcdef_20260923abc',  # short placement
    'sparkrun_0123456789abcdef_20260923abcde',  # long placement
    'sparkrun_3f2be18c41effca0_20260923a1',  # rejected first-launch format
    'sparkrun_0123456789abcdef_20260923abcg',  # non-hex placement
    'sparkrun_0123456789abcdef_20260923ABCd',  # uppercase placement
])
def test_config_rejects_noncanonical_launch_id(config_inputs, launch_id):
    h, recipe, mod, pins = config_inputs
    with pytest.raises(ValueError, match='invalid launch ID'):
        h.make_config(recipe, mod, pins, launch_id, '0123456789abcdef',
                      'vacation-pair2', 'http://127.0.0.1:8000', 'http://127.0.0.1:4000')


def test_config_pins_recipe_mod_source_and_harness(tmp_path, config_inputs):
    import json
    h, recipe, mod, pins = config_inputs
    config = h.make_config(recipe, mod, pins, 'sparkrun_0123456789abcdef_20260923abcd',
                           '0123456789abcdef', 'vacation-pair2', 'http://127.0.0.1:8000', 'http://127.0.0.1:4000')
    assert config['rank_argv']['1'][-1] == '--headless'
    assert config['hosts'] == list(h.HOSTS)
    assert config['mod_files'] == {'sample.py': h.sha(mod / 'sample.py')}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    assert h.load_config(config_path, h.sha(config_path)) == config
    with pytest.raises(ValueError, match='config digest'):
        h.load_config(config_path, '0'*64)
    for bad_version in ('', None, 'wrong'):
        with pytest.raises(ValueError, match='InstantTensor'):
            h.make_config(recipe, mod, {**pins, 'instanttensor_version': bad_version}, config['launch_id'],
                          config['run_id'], config['cluster'], config['direct_url'], config['proxy_url'])
    with pytest.raises(ValueError, match='source'):
        h.make_config(recipe, mod, {**pins, 'source_revision': '0'*40}, config['launch_id'],
                      config['run_id'], config['cluster'], config['direct_url'], config['proxy_url'])
    (mod / 'sample.py').write_text('changed\n')
    with pytest.raises(ValueError, match='mod file'):
        h.load_config(config_path, h.sha(config_path))


def load_component(name):
    assert (HARNESS / (name + '.py')).is_file(), name + ' is missing'
    sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location('refresh_' + name, HARNESS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_command_contract_binds_hosts_rank_interval_and_probe():
    m = load_component('capture_runtime')
    config = {'launch_id': 'sparkrun_0123456789abcdef_20260923abcd', 'env': {'GLM53_MIXED_PREFILL_CHUNK': 'fair'},
              'pins': {'image': 'image@sha256:' + 'a'*64}}
    commands = m.commands(config, 1, 1000.0, 2000.0)
    assert {'identity', 'kernel_boot_before', 'kernel', 'kernel_boot_after', 'serve_log', 'patch_state', 'mod_manifest', 'ready'} == set(commands)
    assert 'journalctl -k -b 0 ' in commands['kernel'][-1]
    assert all(cmd[:4] == ['ssh', '-o', 'BatchMode=yes', '192.168.178.46'] for key, cmd in commands.items() if key != 'ready')
    assert commands['ready'][3] == '192.168.178.47'
    assert config['launch_id'] + '_node_1' in commands['identity'][-1]
    assert "@1000.000000" in commands['kernel'][-1]
    assert "@2000.000000" in commands['kernel'][-1]
    assert '/proc/' in commands['identity'][-1]


def test_cli_help_is_offline_and_no_receipts_created(tmp_path):
    import subprocess
    for name in ['prepare', 'launch_receipt', 'acceptance', 'capture_runtime', 'verify']:
        assert (HARNESS / (name + '.py')).is_file(), name + ' missing'
        result = subprocess.run([sys.executable, str(HARNESS / (name + '.py')), '--help'],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_non_json_http_error_preserves_raw_receipt(monkeypatch):
    import io
    import urllib.error
    from types import SimpleNamespace
    a = load_component('acceptance')
    def fail(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 502, 'bad gateway', {}, io.BytesIO(b'upstream unavailable'))
    monkeypatch.setattr(a.urllib.request, 'build_opener', lambda *args: SimpleNamespace(open=fail))
    row = a.request('http://127.0.0.1:8000/v1/chat/completions', {'test_only': True})
    assert row['http'] == 502
    assert row['response_text'] == 'upstream unavailable'
    assert row['response'] is None
    assert row['request_text'] == '{"test_only": true}'


def test_transport_error_preserves_request_without_inventing_response(monkeypatch):
    from types import SimpleNamespace
    a = load_component('acceptance')
    def fail(req, timeout):
        raise TimeoutError('test-only timeout')
    monkeypatch.setattr(a.urllib.request, 'build_opener', lambda *args: SimpleNamespace(open=fail))
    row = a.request('http://127.0.0.1:8000/v1/chat/completions', {'test_only': True})
    assert row['http'] is None and row['response'] is None and row['response_text'] is None
    assert 'timeout' in row['transport_error']


VIDEO_DIAGNOSTIC = 'glm53: video placeholders aligned to encoder grid_t (Glm5Next→glm46v)\n'
E3_EXECUTED = '[glm53-e3-executed] grouped_calls=1 fat_expert_runs=2 configured_tier=grouped effective_tier=grouped'


def test_only_patch_check_accepts_exact_video_startup_diagnostic(tmp_path):
    import json
    c, digest = unit_bundle(tmp_path)
    v, h = load_component('verify'), load_harness()
    path = tmp_path / 'after.json'
    original = json.loads(path.read_text())
    for host in h.HOSTS:
        original['hosts'][host]['patch_state']['stderr'] = VIDEO_DIAGNOSTIC
    path.write_text(json.dumps(original))
    refresh_unit_bindings(tmp_path, h)
    v.verify_bundle(c, tmp_path, digest)
    for key, text in [('identity', VIDEO_DIAGNOSTIC), ('mod_manifest', VIDEO_DIAGNOSTIC),
                      ('patch_state', VIDEO_DIAGNOSTIC * 2), ('patch_state', VIDEO_DIAGNOSTIC + 'unexpected\n'),
                      ('patch_state', VIDEO_DIAGNOSTIC.replace('grid_t', 'wrong'))]:
        changed = copy.deepcopy(original)
        changed['hosts'][h.HOSTS[0]][key]['stderr'] = text
        path.write_text(json.dumps(changed))
        refresh_unit_bindings(tmp_path, h)
        with pytest.raises(ValueError, match='capture failure'):
            v.verify_bundle(c, tmp_path, digest)


def test_stdlib_probes_are_site_free_and_nested_stderr_is_retained(monkeypatch, capsys):
    import shlex
    from types import SimpleNamespace
    probe, capture = load_component('host_probe'), load_component('capture_runtime')
    c, _ = identity_fixture()
    commands = capture.commands(c, 0, 1., 2.)
    for key in ('identity', 'mod_manifest'):
        argv = shlex.split(shlex.split(commands[key][-1])[0])
        i = argv.index('python3')
        assert argv[i+1:i+3] == ['-S', '-c']
    monkeypatch.setattr(probe.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout='{}', stderr='unexpected nested diagnostic\n'))
    with pytest.raises(ValueError, match='stderr'):
        probe.run(['test-only'])
    assert capsys.readouterr().err == 'unexpected nested diagnostic\n'


def test_actual_setproctitle_preserves_proc_environment_only_with_pin():
    import json
    import os
    import subprocess
    # Run with uv --with setproctitle; never replace the real extension with a stub.
    pytest.importorskip('setproctitle')
    script = '''import json, pathlib, setproctitle
setproctitle.setproctitle('VLLM::Worker_TP0')
env = pathlib.Path('/proc/self/environ').read_bytes().split(b'\\0')
print(json.dumps(b'GLM_ENV_PROBE=preserved' in env))
'''
    for pinned in (False, True):
        env = {k: v for k, v in os.environ.items() if k != 'SPT_NOENV'}
        env['GLM_ENV_PROBE'] = 'preserved'
        if pinned:
            env['SPT_NOENV'] = '1'
        result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, check=True)
        assert json.loads(result.stdout) is pinned


def test_worker_effective_environment_cannot_be_replaced_by_config():
    v = load_component('verify')
    c, identity = identity_fixture()
    for env in ({}, {'GLM53_MIXED_PREFILL_CHUNK': 'fair', 'SPT_NOENV': '0'},
                {'GLM53_MIXED_PREFILL_CHUNK': None, 'SPT_NOENV': None}):
        changed = copy.deepcopy(identity)
        changed['processes'][1]['env'] = env
        with pytest.raises(ValueError, match='effective process environment'):
            v.verify_identity(c, 0, changed)


def identity_fixture():
    c = {'launch_id': 'sparkrun_0123456789abcdef_20260923abcd', 'model': 'test/model',
         'env': {'GLM53_MIXED_PREFILL_CHUNK': 'fair', 'SPT_NOENV': '1'}, 'rank_argv': {'0': ['python3', 'vllm', 'serve']},
         'pins': {'image': 'image@sha256:' + 'a'*64, 'image_id': 'sha256:' + 'b'*64,
                  'nccl_sha256': 'c'*64, 'nccl_version': '2.30.7', 'nccl_path': '/lib/libnccl.so.2',
                  'instanttensor_version': '0.2.0'}}
    p = {'container': c['launch_id'] + '_node_0', 'rank': 0, 'container_id': 'd'*64,
         'boot_id': '11111111-2222-3333-4444-555555555555', 'image': c['pins']['image'],
         'image_id': c['pins']['image_id'], 'image_inspect_id': c['pins']['image_id'],
         'repo_digests': [c['pins']['image']], 'user': 'root',
         'labels': {'sparkrun.cluster_id': c['launch_id'], 'sparkrun.rank': '0',
                    'sparkrun.model': c['model'], 'sparkrun.served_model_name': 'GLM-5.3-Flash-EXL3'},
         'state': {'Running': True, 'Paused': False, 'Restarting': False, 'OOMKilled': False,
                   'Dead': False, 'StartedAt': '2026-09-23T00:00:00+00:00'},
         'security': {'Privileged': False, 'NetworkMode': 'host', 'IpcMode': 'host',
                      'CapAdd': ['CAP_IPC_LOCK'], 'SecurityOpt': ['no-new-privileges', 'label=disable']},
         'nccl_version': '2.30.7', 'instanttensor_version': '0.2.0', 'listeners': [{'inode': '55', 'owners': [10]}],
         'docker_top': 'PID PPID PGID SID COMMAND\n100 1 100 100 python3 vllm serve\n102 100 100 100 VLLM::EngineCore\n101 102 100 100 VLLM::Worker_TP0\n',
         'namespace': [{'host_pid': 100, 'pid': 10, 'nspid': [100,10], 'start_ticks': 20},
                       {'host_pid': 102, 'pid': 12, 'nspid': [102,12], 'start_ticks': 21},
                       {'host_pid': 101, 'pid': 11, 'nspid': [101,11], 'start_ticks': 21}],
         'processes': [{'pid': 10, 'start_ticks': 20, 'argv': c['rank_argv']['0'], 'env': c['env'], 'nccl': []},
                       {'pid': 11, 'start_ticks': 21, 'argv': ['VLLM::Worker_TP0'], 'env': c['env'],
                        'nccl': [{'path': '/lib/libnccl.so.2', 'sha256': 'c'*64}]}]}
    return c, p


def test_runtime_identity_semantics_reject_rank_digest_process_and_security_mutations():
    v = load_component('verify')
    c, p = identity_fixture()
    v.verify_identity(c, 0, p)
    bad_version = copy.deepcopy(p)
    bad_version['instanttensor_version'] = 'wrong'
    with pytest.raises(ValueError, match='InstantTensor'):
        v.verify_identity(c, 0, bad_version)
    mutations = [lambda d: d.update(rank=1), lambda d: d.update(image_id='sha256:'+'0'*64),
                 lambda d: d['labels'].update({'sparkrun.cluster_id': 'other'}),
                 lambda d: d['security'].update(Privileged=True),
                 lambda d: d['processes'][1]['nccl'][0].update(sha256='0'*64),
                 lambda d: d['processes'][0]['env'].update(GLM53_MIXED_PREFILL_CHUNK='skip'),
                 lambda d: d['namespace'][0].update(pid=99),
                 lambda d: d.update(docker_top=d['docker_top'].replace('100 1', '900 1')),
                 lambda d: d['listeners'][0].update(owners=[99]),
                 lambda d: d['state'].update(OOMKilled=True)]
    for mutate in mutations:
        modified = copy.deepcopy(p)
        mutate(modified)
        with pytest.raises(ValueError):
            v.verify_identity(c, 0, modified)


def test_timestamp_and_kernel_checks_fail_closed():
    v = load_component('verify')
    v.verify_interval(10, 11, 9, 12)
    for start, end in [(float('nan'), 11), (11, 10), (8, 11), (10, float('inf'))]:
        with pytest.raises(ValueError):
            v.verify_interval(start, end, 9, 12)
    v.verify_kernel('-- No entries --\n')
    for text in ['NVRM: Xid 31', 'NV_ERR_NO_MEMORY', 'Out of memory: Killed process',
                 'NCCL error: unhandled system error', 'Segmentation fault',
                 'oom-kill: process', 'Permission denied', 'Journal files have been rotated']:
        with pytest.raises(ValueError):
            v.verify_kernel(text)


def unit_bundle(tmp_path):
    """Deliberately synthetic unit data, confined to pytest scratch; not a live receipt."""
    import base64
    import json
    from datetime import datetime, timezone
    h, v, capture = load_harness(), load_component('verify'), load_component('capture_runtime')
    c, identity = identity_fixture()
    c.update(schema=1, run_id='0123456789abcdef', cluster='test-only', recipe='/test/recipe.yaml',
             direct_url='http://127.0.0.1:8000', proxy_url='http://127.0.0.1:4000', mod_files={'test-only': 'e'*64})
    c['pins']['mod_manifest_sha256'] = 'f'*64
    c['rank_argv']['1'] = ['python3', 'vllm', 'serve', '--headless']
    config_sha = '1'*64
    launch = {'schema': 1, 'argv': h.launch_argv(c), 'returncode': 0, 'pins': c['pins'],
              'config_sha256': config_sha, 'launch_id': c['launch_id'], 'run_id': c['run_id'],
              'started_at': 1000., 'completed_at': 1002.}
    (tmp_path / 'launch.log').write_text('TEST-ONLY SYNTHETIC LAUNCH OUTPUT\n')
    def dump(name, data):
        (tmp_path / name).write_text(json.dumps(data))
    launch['launch_log_sha256'] = h.sha(tmp_path / 'launch.log')
    dump('launch.json', launch)
    ready = datetime.fromtimestamp(1005, timezone.utc).isoformat()
    def snapshot(phase, start, end):
        result = {'schema': 1, 'config_sha256': config_sha, 'launch_id': c['launch_id'], 'run_id': c['run_id'],
                  'phase': phase, 'started_at': start, 'completed_at': end, 'readiness_at': 1005.,
                  'window_until': start + 1, 'launch_sha256': h.sha(tmp_path / 'launch.json'), 'hosts': {}}
        for rank, host in enumerate(h.HOSTS):
            p = copy.deepcopy(identity)
            p.update(rank=rank, container=c['launch_id'] + '_node_' + str(rank))
            p['labels']['sparkrun.rank'] = str(rank)
            p['state']['StartedAt'] = datetime.fromtimestamp(1001, timezone.utc).isoformat()
            p['processes'][0]['argv'] = c['rank_argv'][str(rank)]
            p['processes'][1]['argv'] = [f'VLLM::Worker_TP{rank}']
            p['docker_top'] = ('PID PPID PGID SID COMMAND\n100 1 100 100 ' + ' '.join(c['rank_argv'][str(rank)])
                               + ('\n102 100 100 100 VLLM::EngineCore' if rank == 0 else '')
                               + f'\n101 {102 if rank == 0 else 100} 100 100 VLLM::Worker_TP{rank}\n')
            if rank:
                p['listeners'] = []
                p['namespace'] = [row for row in p['namespace'] if row['host_pid'] != 102]
            outputs = {'identity': json.dumps(p), 'ready': ready,
                       'kernel_boot_before': p['boot_id'] + '\n', 'kernel_boot_after': p['boot_id'] + '\n',
                       'kernel': '-- No entries --\n', 'serve_log': 'Using fp8_ds_mla KV cache format\nGraph capturing finished\n' + E3_EXECUTED + '\n',
                       'patch_state': '[OK] complete GLM-5.3 runtime patched state verified\n',
                       'mod_manifest': json.dumps({'manifest_sha256': c['pins']['mod_manifest_sha256'], 'files': c['mod_files']})}
            result['hosts'][host] = {name: {'argv': argv, 'returncode': 0, 'stdout': outputs[name], 'stderr': '',
                                           'started_at': start+2+i*.5, 'completed_at': start+2.25+i*.5}
                                     for i, (name, argv) in enumerate(capture.commands(c, rank, 1005., start+1).items())}
        return result
    before, after = snapshot('before', 1010., 1020.), snapshot('after', 1030., 1040.)
    dump('before.json', before)
    a = {'schema': 1, 'config_sha256': config_sha, 'launch_id': c['launch_id'], 'run_id': c['run_id'],
         'before_sha256': h.sha(tmp_path / 'before.json'), 'started_at': 1021., 'completed_at': 1029., 'checks': {}}
    old = h.legacy_acceptance()
    bodies = h.requests_for_run(c['run_id'])
    for name in ['direct_models', 'proxy_models', *bodies]:
        body = bodies.get(name)
        response = {'object': 'list', 'data': [{'id': h.MODEL}]} if body is None else {
            'object': 'chat.completion', 'model': h.MODEL,
            'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': ''}}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 8, 'total_tokens': 28}}
        status = 200
        if body:
            msg = response['choices'][0]['message']
            choice = response['choices'][0]
            if name.endswith('_rejected'):
                status = 400
                response = getattr(old, ('PROXY_' if name.startswith('proxy') else '') +
                                   ('VIDEO_LIMIT_ERROR' if '_video_' in name else 'REMOTE_MEDIA_ERROR'))
            elif name.endswith('_vision'): msg['content'] = 'RED'
            elif name.endswith('_ocr'): msg['content'] = 'GLM53 OCR 8429'
            elif name == 'direct_long_context':
                msg['content'] = 'NEEDLE_GL53_842917'
                response['usage'].update(prompt_tokens=110030, total_tokens=110038)
            elif name == 'direct_long_output':
                msg['content'] = ' '.join('fixtureword'+str(i) for i in range(2100))
                response['usage'].update(completion_tokens=4096, total_tokens=4116)
                choice['finish_reason'] = 'length'
            elif name == 'direct_tool_call':
                msg['tool_calls'] = [{'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city":"Paris"}'}}]
                choice['finish_reason'] = 'tool_calls'
            elif name == 'direct_tool_none': msg['content'] = 'GLM53_NO_TOOL_OK'
            elif name == 'direct_json_reasoning':
                msg.update(content='{"answer":144,"code":"GLM53_JSON_OK"}', reasoning='12 times 12 is 144')
            elif name == 'direct_reasoning_open_stop':
                msg.update(content='GLM53_REASONING_STOP_OK', reasoning='144')
                choice['stop_reason'] = 154827
            elif name.startswith('direct_thinking_disabled_stop_'):
                msg['content'] = 'BEFORE '
                choice['stop_reason'] = 'Question:'
            elif '_apc_' in name:
                msg['content'] = 'GLM53_APC_OK'
                response['usage']['prompt_tokens_details'] = {'cached_tokens': 10}
            elif name == 'direct_omitted_limit': msg['content'] = 'GLM53_OMITTED_OK'
            elif '_c4_' in name: msg['content'] = 'GLM53_C4_' + name[-1] + '_OK'
            elif name == 'direct_exact': msg['content'] = 'GLM53_DIRECT_OK'
            elif name == 'proxy_exact': msg['content'] = 'GLM53_PROXY_OK'
        url = c['proxy_url' if name.startswith('proxy') else 'direct_url'] + ('/v1/models' if body is None else '/v1/chat/completions')
        row = {'http': status, 'requested_url': url, 'effective_url': url, 'started_at': 1022., 'completed_at': 1023.,
               'response_text': json.dumps(response), 'response': response, 'body_complete': True,
               'response_base64': base64.b64encode(json.dumps(response).encode()).decode('ascii')}
        if name == 'direct_apc_hot': row.update(started_at=1025., completed_at=1026.)
        if body is not None: row.update(request=body, request_text=json.dumps(body), request_sha256=h.request_sha(body))
        a['checks'][name] = row
    dump('acceptance.json', a)
    after['acceptance_sha256'] = h.sha(tmp_path / 'acceptance.json')
    dump('after.json', after)
    refresh_unit_bindings(tmp_path, h)
    return c, config_sha


def refresh_unit_bindings(root, h):
    import json
    a = json.loads((root / 'acceptance.json').read_text())
    a['before_sha256'] = h.sha(root / 'before.json')
    (root / 'acceptance.json').write_text(json.dumps(a))
    after = json.loads((root / 'after.json').read_text())
    after['acceptance_sha256'] = h.sha(root / 'acceptance.json')
    (root / 'after.json').write_text(json.dumps(after))
    (root / 'http.jsonl').write_text(''.join(json.dumps({'name': k, 'receipt': v})+'\n' for k,v in a['checks'].items()))
    (root / 'SHA256SUMS').write_text(''.join(h.sha(root / name)+'  '+name+'\n' for name in sorted(load_component('verify').REQUIRED_FILES) if (root/name).exists()))


def test_acceptance_refuses_bad_runtime_before_any_http(tmp_path, monkeypatch):
    import json
    c, config_sha = unit_bundle(tmp_path)
    for name in ('acceptance.json', 'after.json', 'http.jsonl', 'SHA256SUMS'):
        (tmp_path / name).unlink()
    before = json.loads((tmp_path / 'before.json').read_text())
    before['hosts']['192.168.178.46']['kernel']['stdout'] = 'NV_ERR_NO_MEMORY'
    (tmp_path / 'before.json').write_text(json.dumps(before))
    a = load_component('acceptance')
    monkeypatch.setattr(a, 'load_config', lambda *args: c)
    def no_http(*args, **kwargs):
        pytest.fail('HTTP attempted despite invalid before snapshot')
    monkeypatch.setattr(a, 'request', no_http)
    monkeypatch.setattr(sys, 'argv', ['acceptance.py', '--config', str(tmp_path/'unused.json'),
        '--expected-config-sha256', config_sha, '--out-dir', str(tmp_path)])
    with pytest.raises(ValueError, match='kernel'):
        a.main()
    assert not (tmp_path / 'http.jsonl').exists()


def test_complete_bundle_and_refreshed_manifest_mutations(tmp_path):
    import json
    h, v = load_harness(), load_component('verify')
    c, config_sha = unit_bundle(tmp_path)
    assert set(v.verify_bundle(c, tmp_path, config_sha)) == v.REQUIRED_FILES
    originals = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    def change_acceptance(mutator):
        a = json.loads((tmp_path / 'acceptance.json').read_text())
        mutator(a)
        (tmp_path / 'acceptance.json').write_text(json.dumps(a))
    def change_runtime(mutator):
        a = json.loads((tmp_path / 'after.json').read_text())
        mutator(a)
        (tmp_path / 'after.json').write_text(json.dumps(a))
    cases = [
        lambda: change_acceptance(lambda a: a['checks'].pop('direct_tool_none')),
        lambda: change_acceptance(lambda a: a['checks']['direct_exact'].update(effective_url='http://wrong:8000/v1/chat/completions')),
        lambda: change_acceptance(lambda a: a['checks']['direct_c4_3'].update(started_at=1024., completed_at=1025.)),
        lambda: change_acceptance(lambda a: a.update(started_at=900.)),
        lambda: change_runtime(lambda a: a['hosts'][h.HOSTS[1]]['kernel'].update(stdout='NV_ERR_NO_MEMORY')),
        lambda: change_runtime(lambda a: a['hosts'][h.HOSTS[1]]['kernel']['argv'].__setitem__(3, h.HOSTS[0])),
        lambda: change_runtime(lambda a: a['hosts'][h.HOSTS[0]]['patch_state'].update(returncode=1)),
        lambda: (tmp_path / 'launch.log').unlink(),
    ]
    for mutation in cases:
        for name, raw in originals.items(): (tmp_path / name).write_bytes(raw)
        mutation()
        refresh_unit_bindings(tmp_path, h)
        with pytest.raises(ValueError):
            v.verify_bundle(c, tmp_path, config_sha)


@pytest.mark.parametrize('rank,target,field,value', [
    (rank, target, field, '999') for rank in (0, 1)
    for target, fields in [('worker', (1, 2, 3)), ('serving', (2, 3)), ('engine', (1, 2, 3))]
    if rank == 0 or target != 'engine' for field in fields])
def test_lineage_mutations_fail_with_refreshed_bundle_hashes(tmp_path, rank, target, field, value):
    import json
    c, digest = unit_bundle(tmp_path)
    v, h = load_component('verify'), load_harness()
    v.verify_bundle(c, tmp_path, digest)
    path = tmp_path / 'after.json'
    snapshot = json.loads(path.read_text())
    receipt = snapshot['hosts'][h.HOSTS[rank]]['identity']
    identity = json.loads(receipt['stdout'])
    pid = {'serving': '100', 'worker': '101', 'engine': '102'}[target]
    lines = identity['docker_top'].splitlines()
    for i, line in enumerate(lines[1:], 1):
        fields = line.split(None, 4)
        if fields[0] == pid:
            fields[field] = value
            lines[i] = ' '.join(fields)
    identity['docker_top'] = '\n'.join(lines) + '\n'
    receipt['stdout'] = json.dumps(identity)
    path.write_text(json.dumps(snapshot))
    refresh_unit_bindings(tmp_path, h)
    with pytest.raises(ValueError, match='lineage|group/session'):
        v.verify_bundle(c, tmp_path, digest)


@pytest.mark.parametrize('text', ['arbitrary success\n', '-- No entries --\ntrailing\n',
    '1006.000000 host kernel: benign', 'nan host kernel: benign\n',
    '1004.999999 host kernel: stale\n', '1031.000001 host kernel: future\n',
    '1007.000000 host kernel: fine\n1006.000000 host kernel: reordered\n',
    '1006.000000 host kernel: 12 messages suppressed\n',
    '1006.000000 host kernel: printk: 4 callbacks suppressed\n',
    '1006.000000 host kernel: Missed 10 kernel messages\n',
    '1006.000000 host kernel: journal truncated\n',
    '1006.000000 host kernel: Lost 4 messages\n',
    '1006.000000 host kernel: rate limit exceeded\n'])
def test_kernel_shape_interval_and_loss_mutations(tmp_path, text):
    import json
    c, digest = unit_bundle(tmp_path)
    h, v = load_harness(), load_component('verify')
    v.verify_bundle(c, tmp_path, digest)
    path = tmp_path / 'after.json'
    snapshot = json.loads(path.read_text())
    snapshot['hosts'][h.HOSTS[0]]['kernel']['stdout'] = text
    path.write_text(json.dumps(snapshot))
    refresh_unit_bindings(tmp_path, h)
    with pytest.raises(ValueError, match='kernel'):
        v.verify_bundle(c, tmp_path, digest)


def test_kernel_valid_short_unix_and_no_entries():
    v = load_component('verify')
    v.verify_kernel('-- No entries --\n')
    v.verify_kernel('1006.000000 dgx01 kernel: benign event\n')


@pytest.mark.parametrize('key', ['kernel_boot_before', 'kernel_boot_after'])
def test_kernel_boot_join_rejects_changed_boot_with_fresh_hashes(tmp_path, key):
    import json
    c, digest = unit_bundle(tmp_path)
    h, v = load_harness(), load_component('verify')
    path = tmp_path / 'after.json'
    snapshot = json.loads(path.read_text())
    row = snapshot['hosts'][h.HOSTS[0]].get(key)
    assert row is not None, 'kernel boot capture missing'
    row['stdout'] = '99999999-2222-3333-4444-555555555555\n'
    path.write_text(json.dumps(snapshot))
    refresh_unit_bindings(tmp_path, h)
    with pytest.raises(ValueError, match='kernel boot'):
        v.verify_bundle(c, tmp_path, digest)


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('replacement', ['[glm53-e3-executed]',
    E3_EXECUTED.replace('grouped_calls=1', 'grouped_calls=0'),
    E3_EXECUTED.replace('fat_expert_runs=2', 'fat_expert_runs=0'),
    E3_EXECUTED.replace('configured_tier=grouped', 'configured_tier=disabled'),
    E3_EXECUTED.replace('effective_tier=grouped', 'effective_tier=disabled'),
    E3_EXECUTED.replace('grouped_calls=1', 'grouped_calls=-1'),
    E3_EXECUTED + 'garbage',
    E3_EXECUTED + '\n' + E3_EXECUTED.replace('effective_tier=grouped', 'effective_tier=disabled')])
def test_e3_execution_counters_and_tiers_on_both_ranks(tmp_path, rank, replacement):
    import json
    c, digest = unit_bundle(tmp_path)
    h, v = load_harness(), load_component('verify')
    v.verify_bundle(c, tmp_path, digest)
    path = tmp_path / 'after.json'
    snapshot = json.loads(path.read_text())
    row = snapshot['hosts'][h.HOSTS[rank]]['serve_log']
    row['stdout'] = row['stdout'].replace(E3_EXECUTED, replacement)
    path.write_text(json.dumps(snapshot))
    refresh_unit_bindings(tmp_path, h)
    with pytest.raises(ValueError, match='E3'):
        v.verify_bundle(c, tmp_path, digest)


@pytest.mark.parametrize('failure', ['unicode', 'incomplete', 'timeout', 'bad_json'])
def test_http_body_failures_preserve_raw_status_and_url(monkeypatch, failure):
    import base64
    import http.client
    from types import SimpleNamespace
    a = load_component('acceptance')
    raw = b'\xffraw' if failure == 'unicode' else b'{"partial":'
    class Response:
        code = 200
        def __init__(self): self.calls = 0
        def geturl(self): return 'http://127.0.0.1:8000/v1/models'
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, *args):
            self.calls += 1
            if failure == 'incomplete':
                raise http.client.IncompleteRead(raw, 10)
            if self.calls == 1: return raw
            if failure == 'timeout': raise TimeoutError('test-only body timeout')
            return b''
    monkeypatch.setattr(a.urllib.request, 'build_opener', lambda *args: SimpleNamespace(open=lambda *a, **k: Response()))
    row = a.request('http://127.0.0.1:8000/v1/models')
    assert row['http'] == 200 and row['effective_url'] == row['requested_url']
    assert base64.b64decode(row['response_base64']) == raw
    assert row['response'] is None
    assert row.get('decode_error') if failure in ('unicode', 'bad_json') else row.get('transport_error')
    assert row['completed_at'] >= row['started_at']


def test_failed_http_receipt_journaled_before_semantic_failure(tmp_path, monkeypatch):
    import base64
    import io
    import json
    from types import SimpleNamespace
    a = load_component('acceptance')
    c, digest = unit_bundle(tmp_path)
    for name in ('acceptance.json', 'after.json', 'http.jsonl', 'SHA256SUMS'):
        (tmp_path / name).unlink()
    monkeypatch.setattr(a, 'load_config', lambda *args: c)
    class Response(io.BytesIO):
        code = 200
        def geturl(self): return c['direct_url'] + '/v1/models'
    monkeypatch.setattr(a.urllib.request, 'build_opener', lambda *args:
                        SimpleNamespace(open=lambda *a, **k: Response(b'\xffactual-test-bytes')))
    monkeypatch.setattr(sys, 'argv', ['acceptance.py', '--config', str(tmp_path/'unused.json'),
        '--expected-config-sha256', digest, '--out-dir', str(tmp_path)])
    with pytest.raises(ValueError, match='HTTP receipt'):
        a.main()
    journal = [json.loads(line) for line in (tmp_path / 'http.jsonl').read_text().splitlines()]
    assert len(journal) == 1 and journal[0]['name'] == 'direct_models'
    receipt = journal[0]['receipt']
    assert base64.b64decode(receipt['response_base64']) == b'\xffactual-test-bytes'
    assert receipt['http'] == 200 and receipt['decode_error']
    assert json.loads((tmp_path/'acceptance.json').read_text())['checks']['direct_models'] == receipt


def test_short_content_length_retains_actual_partial_http_bytes(monkeypatch):
    import base64
    import http.client
    import io
    from types import SimpleNamespace
    a = load_component('acceptance')
    wire = b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{}'
    response = http.client.HTTPResponse(SimpleNamespace(makefile=lambda *args: io.BytesIO(wire)))
    response.begin()
    response.code = response.status
    response.geturl = lambda: 'http://127.0.0.1:8000/v1/models'
    monkeypatch.setattr(a.urllib.request, 'build_opener', lambda *args: SimpleNamespace(open=lambda *a, **k: response))
    row = a.request(response.geturl())
    assert row['http'] == 200 and base64.b64decode(row['response_base64']) == b'{}'
    assert not row['body_complete'] and row['transport_error']
    assert row['response'] is None


@pytest.mark.parametrize('mutation', [lambda row: row.update(response_base64='e30='),
    lambda row: row.update(body_complete=False), lambda row: row.update(transport_error='truncated'),
    lambda row: row.update(decode_error='bad utf8'), lambda row: row.pop('response_base64')])
def test_wire_receipt_mutations_fail_with_fresh_hashes(tmp_path, mutation):
    import json
    c, digest = unit_bundle(tmp_path)
    h, v = load_harness(), load_component('verify')
    v.verify_bundle(c, tmp_path, digest)
    path = tmp_path / 'acceptance.json'
    receipt = json.loads(path.read_text())
    mutation(receipt['checks']['direct_exact'])
    path.write_text(json.dumps(receipt))
    refresh_unit_bindings(tmp_path, h)
    with pytest.raises(ValueError, match='HTTP receipt'):
        v.verify_bundle(c, tmp_path, digest)


@pytest.mark.parametrize('key,field,value', [('kernel_boot_before', 'completed_at', 1033.1),
    ('kernel_boot_after', 'started_at', 1033.1), ('kernel', 'started_at', 1030.5)])
def test_kernel_capture_interval_join_fail_closed(tmp_path, key, field, value):
    import json
    c, digest = unit_bundle(tmp_path)
    h, v = load_harness(), load_component('verify')
    v.verify_bundle(c, tmp_path, digest)
    path = tmp_path / 'after.json'
    snapshot = json.loads(path.read_text())
    snapshot['hosts'][h.HOSTS[0]][key][field] = value
    path.write_text(json.dumps(snapshot))
    refresh_unit_bindings(tmp_path, h)
    with pytest.raises(ValueError, match='kernel capture'):
        v.verify_bundle(c, tmp_path, digest)
