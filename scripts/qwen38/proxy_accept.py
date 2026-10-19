#!/usr/bin/env python3
"""Capture LiteLLM discovery, health and semantic chat/media/tool results."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from accept import request_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise SystemExit('refusing to overwrite receipts')
    for name, path in [('models', '/v1/models'), ('health', '/health'),
                       ('chat', '/v1/chat/completions'), ('vision', '/v1/chat/completions'),
                       ('tools', '/v1/chat/completions')]:
        body = request_for(name) if name in ('chat', 'vision', 'tools') else None
        url = 'http://127.0.0.1:4000' + path
        request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        receipt = {'case': name, 'url': url, 'request': body, 'status': status,
                   'started_at': start, 'finished_at': time.time(),
                   'raw_response_base64': base64.b64encode(raw).decode(), 'response': None}
        target = args.output / (name + '.json')
        target.write_text(json.dumps(receipt, indent=2) + '\n')
        receipt['response'] = json.loads(raw)
        target.write_text(json.dumps(receipt, indent=2) + '\n')
        print(name, status, flush=True)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
