#!/usr/bin/env python3
"""Real HTTP acceptance. Never generates substitute responses or performance claims."""
import base64
import concurrent.futures
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from contracts import (common_parser, load_config, require, request_sha, requests_for_run, verify_http_receipt,
                       sha, verify_response, write_json)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, body=None, timeout=3600):
    raw = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=raw, headers={'Content-Type': 'application/json'})
    start = time.time()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    row = {'http': None, 'requested_url': url, 'effective_url': None, 'started_at': start,
           'response_text': None, 'response': None, 'response_base64': None, 'body_complete': False}
    chunks = []
    if raw is not None:
        row.update(request=body, request_text=raw.decode(), request_sha256=request_sha(body))
    try:
        try:
            response = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            # Capture headers/status before any body read can fail.
            row.update(http=response.code, effective_url=response.geturl())
            headers = getattr(response, 'headers', {})
            length = headers.get('Content-Length')
            read_body = getattr(response, 'read1', response.read)
            while True:
                chunk = read_body(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            if length is not None and (not length.isdecimal() or int(length) != sum(map(len, chunks))):
                row['transport_error'] = 'incomplete/invalid Content-Length: ' + length
            else:
                row['body_complete'] = True
    except http.client.IncompleteRead as exc:
        chunks.append(exc.partial)
        row['transport_error'] = str(exc)
    except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
        row['transport_error'] = str(exc)
    if row['http'] is not None:
        received = b''.join(chunks)
        row['response_base64'] = base64.b64encode(received).decode('ascii')
        try:
            row['response_text'] = received.decode('utf-8')
            if row['body_complete']:
                row['response'] = json.loads(row['response_text'])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            row['decode_error'] = str(exc)
    row['completed_at'] = time.time()
    return row


def main():
    a = common_parser(__doc__).parse_args()
    c = load_config(a.config, a.expected_config_sha256)
    before_path = a.out_dir / 'before.json'
    before = json.loads(before_path.read_text())
    require(before['config_sha256'] == a.expected_config_sha256 and before['phase'] == 'before', 'before capture binding')
    from verify import verify_snapshot
    launch_path = a.out_dir / 'launch.json'
    launch = json.loads(launch_path.read_text())
    require(before['launch_sha256'] == sha(launch_path), 'before launch binding')
    verify_snapshot(c, before, 'before', launch, a.expected_config_sha256)
    require(not (a.out_dir / 'acceptance.json').exists() and not (a.out_dir / 'http.jsonl').exists(), 'refuse overwrite')
    record = {'schema': 1, 'launch_id': c['launch_id'], 'run_id': c['run_id'],
              'config_sha256': a.expected_config_sha256, 'before_sha256': sha(before_path),
              'started_at': time.time(), 'checks': {}}
    bodies = requests_for_run(c['run_id'])
    lock = threading.Lock()
    with (a.out_dir / 'http.jsonl').open('x') as journal:
        def collect(name, body=None, barrier=None):
            if barrier:
                barrier.wait(timeout=30)
            url = c['proxy_url' if name.startswith('proxy') else 'direct_url']
            row = request(url + ('/v1/models' if body is None else '/v1/chat/completions'), body)
            with lock:
                journal.write(json.dumps({'name': name, 'receipt': row}) + '\n')
                journal.flush()
                record['checks'][name] = row
            verify_http_receipt(name, row)
            if body is not None:
                verify_response(name, row, body)
            return row
        try:
            for route in ('direct', 'proxy'):
                record['checks'][route + '_models'] = collect(route + '_models')
            group = [f'direct_c4_{i}' for i in range(4)]
            barrier = threading.Barrier(4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {name: pool.submit(collect, name, bodies[name], barrier) for name in group}
                for name, future in futures.items():
                    record['checks'][name] = future.result()
            for name, body in bodies.items():
                if name not in group:
                    record['checks'][name] = collect(name, body)
        finally:
            record['completed_at'] = time.time()
            write_json(a.out_dir / 'acceptance.json', record)
    print(json.dumps({'captured_checks': len(record['checks']), 'semantic_validation': 'run verify.py'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
