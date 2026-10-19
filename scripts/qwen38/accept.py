#!/usr/bin/env python3
"""Small real-HTTP semantic suite; preserves every request and response, no mocks.

Run against the accepted process, then use verify.py on the resulting directory.
This is not a performance benchmark or a near-context-limit qualification.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib

MODEL = 'Qwen3.8-27B'


def image():
    width, height = 384, 192
    raw = b''.join(b'\x00' + b'\xff\x00\x00' * (width // 2) +
                   b'\x00\x00\xff' * (width // 2) for _ in range(height))
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def request_for(case):
    controls = {'model': MODEL, 'temperature': 0, 'seed': 42, 'max_tokens': 256,
                'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}
    if case == 'chat':
        controls['messages'] = [{'role': 'user', 'content': 'Reply with exactly SPARK_READY and nothing else.'}]
    elif case.startswith('concurrent-'):
        index = int(case.split('-')[1])
        controls['messages'] = [{'role': 'user', 'content': f'Reply with exactly SPARK_C{index}_OK and nothing else.'}]
    elif case == 'vision':
        controls['messages'] = [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image()).decode()}},
            {'type': 'text', 'text': 'What color is the left half, and what color is the right half? Answer left and right explicitly.'}]}]
    elif case == 'long':
        lines = [f'Record {i:05d}: ordinary archive entry with no special access code.\n' for i in range(2300)]
        lines.insert(1373, 'The unique archive access code is SPARK_NEEDLE_7K2M.\n')
        controls['messages'] = [{'role': 'user', 'content': ''.join(lines) +
                                 '\nWhat is the unique archive access code? Reply with the code only.'}]
    elif case == 'tools':
        controls['messages'] = [{'role': 'user', 'content': 'Call get_weather for Berlin using unit celsius. Do not answer in prose.'}]
        controls['tools'] = [{'type': 'function', 'function': {'name': 'get_weather',
            'description': 'Get weather for a city.', 'parameters': {'type': 'object',
                'properties': {'city': {'type': 'string'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}},
                'required': ['city', 'unit'], 'additionalProperties': False}}}]
        controls['tool_choice'] = {'type': 'function', 'function': {'name': 'get_weather'}}
    else:
        raise ValueError(case)
    return controls


CASES = ['chat', *[f'concurrent-{i}' for i in range(8)], 'vision', 'tools', 'long']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob('*.json')):
        raise SystemExit('refusing to overwrite existing receipts')

    def capture(name, path, body=None, barrier=None):
        request = urllib.request.Request(args.base.rstrip('/') + path,
            data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        if barrier:
            barrier.wait()
        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=1200) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        receipt = {'case': name, 'url': request.full_url, 'request': body, 'status': status,
                   'started_at': start, 'finished_at': time.time(),
                   'raw_response_base64': base64.b64encode(raw).decode(), 'response': None}
        target = args.output / (name + '.json')
        target.write_text(json.dumps(receipt, indent=2) + '\n')
        receipt['response'] = json.loads(raw)
        target.write_text(json.dumps(receipt, indent=2) + '\n')
        print(name, status, flush=True)
        return receipt

    capture('models-before', '/v1/models')
    capture('health-before', '/health')
    capture('chat', '/v1/chat/completions', request_for('chat'))
    barrier = threading.Barrier(8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda case: capture(case, '/v1/chat/completions', request_for(case), barrier), CASES[1:9]))
    for case in CASES[9:]:
        capture(case, '/v1/chat/completions', request_for(case))
    capture('health-after', '/health')
    capture('models-after', '/v1/models')
    (args.output / 'fixture.png').write_bytes(image())
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
