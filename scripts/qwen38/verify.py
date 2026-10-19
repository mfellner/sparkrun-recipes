#!/usr/bin/env python3
"""Fail-closed semantic verification of accept.py receipts; never calls inference."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accept import CASES, MODEL, image, request_for

REQUIRED = {case + '.json' for case in CASES} | {
    'models-before.json', 'models-after.json', 'health-before.json', 'health-after.json', 'fixture.png'}


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert set(manifest) == REQUIRED, 'manifest must enumerate canonical receipt set'
    assert image() == (root / 'fixture.png').read_bytes(), 'fixture substitution'
    for name, digest in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    receipts = {name[:-5]: json.loads((root / name).read_text()) for name in REQUIRED if name.endswith('.json')}
    bases = set()
    for case, receipt in receipts.items():
        assert receipt['case'] == case and receipt['status'] == 200, case
        assert isinstance(receipt['response'], dict) and 'error' not in receipt['response'], case
        if 'raw_response_base64' in receipt:
            assert json.loads(base64.b64decode(receipt['raw_response_base64'], validate=True)) == receipt['response']
        assert 0 < receipt['started_at'] < receipt['finished_at'], case
        u = urlsplit(receipt['url'])
        assert u.scheme == 'http' and u.hostname and not u.username and not u.password
        bases.add((u.scheme, u.netloc))
        expected_path = '/v1/chat/completions' if case in CASES else '/health' if case.startswith('health') else '/v1/models'
        assert u.path == expected_path and not u.query, case
        if case in CASES:
            assert receipt['request'] == request_for(case), 'request controls changed: ' + case
            body = receipt['response']
            assert body['model'] == MODEL and body['object'] == 'chat.completion', case
            assert body['usage']['prompt_tokens'] > 0 and body['usage']['completion_tokens'] > 0, case
            choices = body['choices']
            assert len(choices) == 1 and choices[0]['finish_reason'] in ('stop', 'tool_calls'), case
            msg = choices[0]['message']
            content = (msg.get('content') or '').strip()
            if case == 'chat':
                assert content == 'SPARK_READY', content
            elif case.startswith('concurrent-'):
                assert content == 'SPARK_C' + case.split('-')[1] + '_OK', content
            elif case == 'long':
                assert content == 'SPARK_NEEDLE_7K2M' and body['usage']['prompt_tokens'] >= 30000, content
            elif case == 'vision':
                lower = content.lower()
                assert all(word in lower for word in ['left', 'right', 'red', 'blue']), content
                assert lower.index('red') < lower.index('blue'), content
            elif case == 'tools':
                calls = msg['tool_calls']
                assert len(calls) == 1 and calls[0]['type'] == 'function'
                fn = calls[0]['function']
                assert fn['name'] == 'get_weather'
                assert json.loads(fn['arguments']) == {'city': 'Berlin', 'unit': 'celsius'}, fn
        else:
            assert receipt['request'] is None
            body = receipt['response']
            if case.startswith('models'):
                assert body['object'] == 'list' and any(m['id'] == MODEL for m in body['data'])
            else:
                assert body['ok'] is True and body['backend'] == 'tensorfold'
    assert len(bases) == 1, 'mixed endpoints'
    concurrent = [receipts[f'concurrent-{i}'] for i in range(8)]
    assert max(r['started_at'] for r in concurrent) < min(r['finished_at'] for r in concurrent), 'HTTP intervals do not overlap'
    assert receipts['health-after']['response']['busy'] is False
    assert receipts['health-after']['response']['requests_running'] == 0
    return {'verdict': 'PASS', 'semantic_cases': len(CASES),
            'long_prompt_tokens': receipts['long']['response']['usage']['prompt_tokens'],
            'client_concurrency': len(concurrent), 'endpoint': next(iter(bases))[1]}


def negative_controls(root):
    import shutil
    import tempfile
    # Copies stay below the supplied evidence directory's parent, never system /tmp.
    def rejected(change):
        with tempfile.TemporaryDirectory(prefix='qwen38-negative-', dir=root.parent) as tmp:
            p = Path(tmp) / 'receipts'
            shutil.copytree(root, p)
            change(p)
            try:
                verify(p)
            except (AssertionError, KeyError, ValueError, FileNotFoundError):
                return
            raise AssertionError('negative control unexpectedly passed')
    def mutate(p, field, value):
        file = p / 'chat.json'
        data = json.loads(file.read_text())
        if field == 'status':
            data['status'] = value
        elif field == 'model':
            data['response']['model'] = value
        elif field == 'request':
            data['request']['temperature'] = value
        if 'raw_response_base64' in data:
            data['raw_response_base64'] = base64.b64encode(json.dumps(data['response']).encode()).decode()
        file.write_text(json.dumps(data))
        manifest = json.loads((p / 'manifest.json').read_text())
        manifest['chat.json'] = hashlib.sha256(file.read_bytes()).hexdigest()
        (p / 'manifest.json').write_text(json.dumps(manifest))
    rejected(lambda p: mutate(p, 'status', 500))
    rejected(lambda p: mutate(p, 'model', 'wrong-backend'))
    rejected(lambda p: mutate(p, 'request', 1))
    def delete_receipt(p):
        (p / 'tools.json').unlink()
        manifest = json.loads((p / 'manifest.json').read_text())
        del manifest['tools.json']
        (p / 'manifest.json').write_text(json.dumps(manifest))
    rejected(delete_receipt)
    return 4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--negative-controls', action='store_true')
    args = parser.parse_args()
    result = verify(args.directory)
    if args.negative_controls:
        result['negative_controls_rejected'] = negative_controls(args.directory)
    print(json.dumps(result, indent=2))
