#!/usr/bin/env python3
"""Reviewed, parameterized contracts; no live IDs or receipt hashes baked in."""
from __future__ import annotations

import hashlib
import base64
import binascii
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HISTORICAL = ROOT.with_name('glm53-exl3-850k-20260913')
HOSTS = ('192.168.178.47', '192.168.178.46')
MODEL = 'GLM-5.3-Flash-EXL3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_module(name, expected):
    path = HISTORICAL / (name + '.py')
    require(sha(path) == expected, 'historical helper changed: ' + name)
    spec = importlib.util.spec_from_file_location('glm53_historical_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_acceptance():
    return historical_module('acceptance', '13db35afcc47b7aff8ed0166a8eb32b08867a2cc5af4d13384b6d15c1e82cc1c')


def command_contract():
    return historical_module('command_contract', 'c00e8c9b4cf6a752eda1380b74bcabb6aaf1c02c17d9b7f6d719cecf1cab3eaa')


def request_sha(body):
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


def requests_for_run(run_id):
    require(re.fullmatch('[0-9a-f]{16}', run_id), 'invalid run ID')
    old = legacy_acceptance()
    bodies = {'direct_exact': old.exact_request('GLM53_DIRECT_OK'),
              'proxy_exact': old.exact_request('GLM53_PROXY_OK'),
              'direct_reasoning_open_stop': old.reasoning_open_stop_request()}
    for i in range(10):
        stop = old.thinking_disabled_stop_request()
        stop.update(ignore_eos=True, structured_outputs={'regex': 'BEFORE Question: AFTER'})
        bodies[f'direct_thinking_disabled_stop_{i}'] = stop
    for i in range(4):
        bodies[f'direct_c4_{i}'] = old.exact_request(f'GLM53_C4_{i}_OK')
    for route in ('direct', 'proxy'):
        bodies[route + '_vision'] = old.vision_request(old.make_quadrant_png())
        bodies[route + '_ocr'] = old.ocr_request(old.make_ocr_png())
        bodies[route + '_remote_media_rejected'] = old.remote_media_request()
        bodies[route + '_video_rejected'] = old.video_rejection_request(old.make_video_fixture())
    long = old.exact_request('NEEDLE_GL53_842917')
    filler = 'alpha ' * 55000
    long['messages'][0]['content'] = filler + '\nHidden retrieval code: NEEDLE_GL53_842917\n' + filler + '\nReply with exactly NEEDLE_GL53_842917.'
    bodies['direct_long_context'] = long
    tool = old.exact_request('unused')
    tool.update(max_tokens=128, tool_choice='auto', tools=[{'type': 'function', 'function': {
        'name': 'get_weather', 'description': 'Get current weather for a city',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}])
    tool['messages'][0]['content'] = 'Use the weather tool for Paris, France.'
    bodies['direct_tool_call'] = tool
    none = json.loads(json.dumps(tool))
    none['tool_choice'] = 'none'
    none['messages'][0]['content'] = 'Do not call tools. Reply with exactly GLM53_NO_TOOL_OK and nothing else.'
    bodies['direct_tool_none'] = none
    omitted = old.exact_request('GLM53_OMITTED_OK')
    del omitted['max_tokens']
    bodies['direct_omitted_limit'] = omitted
    output = old.exact_request('unused')
    output.update(max_tokens=4096, ignore_eos=True)
    output['messages'][0]['content'] = ('Write a detailed technical guide to implementing a reliable distributed job queue. '
        'Discuss persistence, retries, idempotency, ordering, leases, timeouts, backpressure, and recovery. '
        'Use at least 3000 words. Continue until the token budget is exhausted.')
    bodies['direct_long_output'] = output
    structured = old.exact_request('unused')
    structured.update(max_tokens=2300, chat_template_kwargs={'enable_thinking': True}, response_format={
        'type': 'json_schema', 'json_schema': {'name': 'arithmetic', 'strict': True, 'schema': {
            'type': 'object', 'properties': {'answer': {'type': 'integer', 'enum': [144]},
            'code': {'type': 'string', 'enum': ['GLM53_JSON_OK']}},
            'required': ['answer', 'code'], 'additionalProperties': False}}})
    structured['messages'][0]['content'] = 'Compute 12 times 12. Return JSON with answer and code GLM53_JSON_OK.'
    bodies['direct_json_reasoning'] = structured
    apc = old.exact_request('GLM53_APC_OK')
    apc['messages'][0]['content'] = f'Run {run_id}\n' + 'cache prefix fact. ' * 4096 + '\nReply with exactly GLM53_APC_OK and nothing else.'
    bodies['direct_apc_cold'] = apc
    bodies['direct_apc_hot'] = json.loads(json.dumps(apc))
    return bodies


def verify_http_receipt(name, row):
    require(row.get('body_complete') is True and not row.get('decode_error') and not row.get('transport_error')
            and isinstance(row.get('response'), dict) and isinstance(row.get('response_base64'), str),
            name + ': failed HTTP receipt')
    try:
        raw = base64.b64decode(row['response_base64'], validate=True)
        require(raw.decode('utf-8') == row['response_text'] and json.loads(raw) == row['response'],
                name + ': HTTP receipt raw/text/JSON mismatch')
    except (UnicodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise ValueError(name + ': invalid HTTP receipt encoding') from exc


def verify_response(name, row, body):
    require(row['request'] == body, name + ': canonical request')
    require(row['request_sha256'] == request_sha(body), name + ': request digest')
    response = row['response']
    if name.endswith('_rejected'):
        old = legacy_acceptance()
        expected = getattr(old, ('PROXY_' if name.startswith('proxy') else '') +
                           ('VIDEO_LIMIT_ERROR' if '_video_' in name else 'REMOTE_MEDIA_ERROR'))
        require(row['http'] == 400 and response == expected, name + ': exact rejection')
        return
    require(row['http'] == 200 and 'error' not in response, name + ': HTTP/error')
    require(response['model'] == MODEL and response['object'] == 'chat.completion', name + ': response identity')
    require(len(response['choices']) == 1, name + ': choices')
    choice = response['choices'][0]
    message = choice['message']
    require(message['role'] == 'assistant', name + ': role')
    require(choice['finish_reason'] in ('stop', 'length', 'tool_calls'), name + ': finish reason')
    usage = response['usage']
    require(all(type(usage[k]) is int and usage[k] > 0 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')), name + ': usage')
    require(usage['prompt_tokens'] + usage['completion_tokens'] == usage['total_tokens'], name + ': usage sum')
    require(usage['completion_tokens'] <= body.get('max_tokens', 65536), name + ': completion bound')
    content = message.get('content') or ''
    exact = {'direct_exact': 'GLM53_DIRECT_OK', 'proxy_exact': 'GLM53_PROXY_OK',
             'direct_long_context': 'NEEDLE_GL53_842917', 'direct_tool_none': 'GLM53_NO_TOOL_OK',
             'direct_omitted_limit': 'GLM53_OMITTED_OK', 'direct_apc_cold': 'GLM53_APC_OK',
             'direct_apc_hot': 'GLM53_APC_OK', 'direct_reasoning_open_stop': 'GLM53_REASONING_STOP_OK'}
    exact.update({f'direct_c4_{i}': f'GLM53_C4_{i}_OK' for i in range(4)})
    exact.update({f'{route}_{kind}': value for route in ('direct', 'proxy')
                  for kind, value in [('vision', 'RED'), ('ocr', 'GLM53 OCR 8429')]})
    if name in exact:
        require(content.strip() == exact[name] and choice['finish_reason'] == 'stop', name + ': exact content')
    if name != 'direct_tool_call':
        require(not message.get('tool_calls'), name + ': unexpected tool calls')
    if name == 'direct_tool_call':
        calls = message.get('tool_calls') or []
        require(len(calls) == 1 and choice['finish_reason'] == 'tool_calls', name + ': tool count/finish')
        call = calls[0]
        require(call['type'] == 'function' and call['function']['name'] == 'get_weather', name + ': tool identity')
        args = json.loads(call['function']['arguments'])
        require(set(args) == {'city'} and args['city'] in ('Paris', 'Paris, France'), name + ': tool arguments')
    if name == 'direct_long_context':
        require(110000 <= usage['prompt_tokens'] <= 112000, name + ': measured 110K tier')
    if name == 'direct_long_output':
        require(usage['completion_tokens'] > 2000 and len(content) > 6000 and len(set(content.split())) > 100,
                name + ': long visible output')
        require(choice['finish_reason'] == 'length', name + ': output limit')
    if name == 'direct_json_reasoning':
        require(json.loads(content) == {'answer': 144, 'code': 'GLM53_JSON_OK'}, name + ': JSON semantics')
        require(bool(message.get('reasoning')) and choice['finish_reason'] == 'stop', name + ': reasoning')
    if name == 'direct_reasoning_open_stop':
        require('144' in (message.get('reasoning') or '') and choice.get('stop_reason') == 154827, name + ': reasoning stop')
    if name.startswith('direct_thinking_disabled_stop_'):
        require(content == 'BEFORE ' and choice['finish_reason'] == 'stop' and choice.get('stop_reason') == 'Question:', name + ': literal stop')
    if name == 'direct_apc_hot':
        require(usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) > 0, name + ': APC cache hit')


def mod_entries(mod):
    entries = {}
    mod = Path(mod).resolve()
    for line in (mod / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(re.fullmatch('[0-9a-f]{64}', digest), 'invalid mod digest')
        require(name not in entries and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'invalid mod path')
        target = (mod / name).resolve()
        require(target.is_relative_to(mod) and sha(target) == digest, 'mod file: ' + name)
        entries[name] = digest
    require(bool(entries), 'empty mod manifest')
    return entries


def make_config(recipe, mod, pins, launch_id, run_id, cluster, direct_url, proxy_url):
    import yaml
    contract = command_contract()
    recipe, mod = Path(recipe).resolve(), Path(mod).resolve()
    require(sha(recipe) == pins['recipe_sha256'], 'recipe digest')
    require(sha(mod / 'SHA256SUMS') == pins['mod_manifest_sha256'], 'mod manifest digest')
    for key in ('recipe_sha256', 'mod_manifest_sha256', 'nccl_sha256'):
        require(re.fullmatch('[0-9a-f]{64}', pins[key]), 'invalid pin: ' + key)
    require(isinstance(pins.get('instanttensor_version'), str)
            and re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', pins['instanttensor_version']), 'invalid InstantTensor version pin')
    require(re.fullmatch('[0-9a-f]{40}', pins['source_revision']), 'invalid source revision')
    require(re.fullmatch(r'[^\s@]+@sha256:[0-9a-f]{64}', pins['image']), 'unpinned image')
    require(re.fullmatch('sha256:[0-9a-f]{64}', pins['image_id']), 'invalid image config digest')
    # SparkRun 0.3.6 job_metadata: 16-hex intent + 12-hex placement token.
    require(re.fullmatch(r'sparkrun_[0-9a-f]{16}_[0-9a-f]{12}', launch_id), 'invalid launch ID')
    require(re.fullmatch('[0-9a-f]{16}', run_id), 'invalid run ID')
    require(re.fullmatch('[a-zA-Z0-9_-]+', cluster), 'invalid cluster')
    require(direct_url in ('http://127.0.0.1:8000', 'http://192.168.178.47:8000'), 'direct URL must reach rank 0')
    require(proxy_url in ('http://127.0.0.1:4000', 'http://192.168.178.47:4000'), 'proxy URL must reach reviewed LAN proxy')
    data = yaml.load(recipe.read_text(), Loader=contract.UniqueKeyLoader)
    require(data['metadata']['source_revision'] == pins['source_revision'], 'recipe source revision')
    require(data['container'] == pins['image'], 'recipe image')
    argv = contract.reviewed_rank_argv(recipe)
    env = {key: str(value) for key, value in data['env'].items()}
    validate_defaults(env, argv[0])
    require(all(not re.search(r'PASSWORD|SECRET|API_KEY|ACCESS_TOKEN', key) for key in env), 'credential env not supported')
    return {'schema': 1, 'pins': pins, 'recipe': str(recipe), 'mod_dir': str(mod),
            'mod_files': mod_entries(mod), 'source_revision': pins['source_revision'],
            'launch_id': launch_id, 'run_id': run_id, 'cluster': cluster, 'model': data['model'],
            'hosts': list(HOSTS), 'direct_url': direct_url, 'proxy_url': proxy_url,
            'rank_argv': {str(rank): value for rank, value in argv.items()}, 'env': env,
            'harness_sha256': {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}}


def load_config(path, expected):
    require(re.fullmatch('[0-9a-f]{64}', expected) and sha(path) == expected, 'config digest')
    config = json.loads(Path(path).read_text())
    derived = make_config(config['recipe'], config['mod_dir'], config['pins'], config['launch_id'],
                          config['run_id'], config['cluster'], config['direct_url'], config['proxy_url'])
    require(config == derived, 'config no longer matches reviewed inputs/harness')
    return config


def launch_argv(config):
    return ['sparkrun', 'run', config['recipe'], '--cluster', config['cluster'], '--trust',
            '--no-follow', '--container-name', config['launch_id']]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def common_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    return parser


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_defaults(env, argv):
    require(env.get('SPT_NOENV') == '1', 'SPT_NOENV must preserve real worker /proc environment')
    require(env.get('GLM53_MIXED_PREFILL_CHUNK') == 'fair', 'scheduler must be fair')
    require(str(env.get('GLM53_DRAFT_KV_COMPACT')) == '0', 'compact draft must remain off')
    require(str(env.get('DEFAULT_MAX_NEW_TOKENS')) == '65536', 'bounded omitted completion default must be 65536')
    for flag, value in {'--load-format': 'instanttensor', '--kv-cache-memory-bytes': '15032385536',
                        '--tensor-parallel-size': '2', '--max-model-len': '850000',
                        '--max-num-seqs': '4', '--allowed-media-domains': 'media.invalid'}.items():
        require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
                and argv[argv.index(flag) + 1] == value, 'reviewed default: ' + flag)
    require(argv.count('--enable-prefix-caching') == 1, 'APC must be enabled')
    require(argv.count('--limit-mm-per-prompt') == 1, 'missing multimodal limits')
    require(json.loads(argv[argv.index('--limit-mm-per-prompt') + 1]) == {'image': 4, 'video': 0}, 'image-only boundary')
