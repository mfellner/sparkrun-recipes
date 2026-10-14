#!/usr/bin/env python3
"""Matched real HTTP performance collection and strict accounting. SPDX-License-Identifier: MIT."""
import math
from pathlib import Path
import struct
import argparse
import base64
import hashlib
import json
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import request_for


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identity(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def harness_hashes():
    root = Path(__file__).resolve().parent
    return {name: digest((root/name).read_bytes()) for name in ('performance.py', 'fixtures.py', 'acceptance.py')}


def audio_metrics(rate, samples, wall_s):
    if type(rate) is not int or type(samples) is not int or rate <= 0 or samples <= 0:
        raise ValueError('positive integral sample count and rate required')
    if type(wall_s) not in (float, int) or not math.isfinite(wall_s) or wall_s <= 0:
        raise ValueError('positive finite wall time required')
    duration = samples / rate
    return dict(duration_s=duration, rtf=wall_s / duration)


def pcm_offset(raw):
    if len(raw) < 12:
        return None
    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise ValueError('not RIFF WAV')
    pos = 12
    valid_format = False
    while pos + 8 <= len(raw):
        name, size = raw[pos:pos+4], struct.unpack_from('<I', raw, pos+4)[0]
        if name == b'data':
            if not valid_format:
                raise ValueError('PCM format chunk required before data')
            return pos + 8
        if pos + 8 + size > len(raw):
            return None
        if name == b'fmt ':
            if size < 16:
                raise ValueError('short PCM format')
            codec, channels, rate, byte_rate, align, bits = struct.unpack_from('<HHIIHH', raw, pos+8)
            if (codec, channels, rate, byte_rate, align, bits) != (1, 1, 44100, 88200, 2, 16):
                raise ValueError('expected mono 44100Hz PCM16')
            valid_format = True
        pos += 8 + size + size % 2
        if pos > 65536:
            raise ValueError('oversized WAV metadata')
    return None


def first_pcm(raw, events):
    offset = pcm_offset(raw)
    if offset is None:
        raise ValueError('incomplete WAV metadata')
    total, last, first = 0, -1, None
    for event in events:
        elapsed, size = event['elapsed_s'], event['bytes']
        if type(size) is not int or size <= 0 or not math.isfinite(elapsed) or elapsed < last or elapsed <= 0:
            raise ValueError('invalid transport event ordering')
        last = elapsed
        total += size
        if first is None and total >= offset + 2:
            first = elapsed
    if total != len(raw) or first is None:
        raise ValueError('incomplete PCM transport')
    return first


def safe_output(output, inputs):
    target = Path(output).resolve()
    if target in {Path(p).resolve() for p in inputs} or target.exists():
        raise ValueError('input/output collision or existing output')
    return target


def collect(endpoint, directory, fixture, name, arm, runtime, warmup=False, seed=42):
    from acceptance import inspect_audio
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inputs = [directory/'reference-input.wav', directory/'reference.txt']
    output = safe_output(directory/(name+'.wav'), inputs)
    receipt_file = safe_output(directory/(name+'.json'), inputs)
    request = request_for(fixture)
    request['seed'] = seed
    reference_raw = b''
    if fixture == 'reference':
        reference_raw = inputs[0].read_bytes()
        request['references'] = [dict(audio=base64.b64encode(reference_raw).decode(), text=inputs[1].read_text().strip(), consent=True)]
    sent = json.dumps(request).encode()
    recorded = json.loads(sent)
    if reference_raw:
        recorded['references'][0]['audio'] = 'sha256:'+digest(reference_raw)
    receipt = dict(schema=1, arm=arm, fixture=fixture, name=name, warmup=warmup, endpoint=endpoint, route='/v1/tts', request=recorded, request_identity=identity(recorded), runtime=runtime, runtime_identity=identity(runtime), harness=harness_hashes(), complete=False, started_at=time.time(), events=[])
    parts = []
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(endpoint+'/v1/tts', sent, {'Content-Type': 'application/json'}), timeout=1800) as response:
            receipt['status'] = response.status
            receipt['headers'] = {k.lower(): v for k, v in response.headers.items() if k.lower() in {'content-type', 'content-length', 'transfer-encoding', 'x-request-id', 'x-model-revision', 'date', 'server', 'x-stream-protocol'}}
            receipt['headers_s'] = time.perf_counter()-start
            while part := response.read1(65536):
                parts.append(part)
                receipt['events'].append(dict(elapsed_s=time.perf_counter()-start, bytes=len(part)))
        receipt['wall_s'] = time.perf_counter()-start
        raw = b''.join(parts)
        audio = inspect_audio(raw)
        receipt.update(audio=audio, audio_file=output.name, audio_sha256=digest(raw), first_pcm_s=first_pcm(raw, receipt['events']), **audio_metrics(audio['sample_rate'], audio['samples'], receipt['wall_s']))
        receipt['request_id'] = receipt['headers'].get('x-request-id')
        if receipt['status'] != 200 or receipt['headers'].get('x-model-revision') != 'dd64a85cd82e9521f7202610ede97b367fc83946':
            raise ValueError('HTTP status or model identity mismatch')
        if not receipt['request_id'] or len(receipt['request_id']) != 32:
            raise ValueError('request ID absent')
        receipt['complete'] = True
        return receipt
    except Exception as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt['finished_at'] = time.time()
        output.write_bytes(b''.join(parts))
        receipt_file.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k: receipt[k] for k in ('name', 'arm', 'complete', 'wall_s', 'first_pcm_s', 'rtf', 'error') if k in receipt}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='http://192.168.178.70:8010')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--runtime-json', type=Path, required=True)
    parser.add_argument('--fixtures', default='english-1,english-2,german-1,expression,longer,streaming')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error('primary qualification requires at least five repetitions')
    runtime = json.loads(args.runtime_json.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(arm=args.arm, runtime=runtime, harness=harness_hashes(), fixtures=args.fixtures.split(','), repeats=args.repeats, seed=args.seed, warmups=1, order='round-robin fixtures, warmup then measured repetitions', configured_at=time.time())
    safe_output(args.output/'schedule.json', []).write_text(json.dumps(config, indent=2)+'\n')
    for repetition in range(args.repeats+1):
        for fixture in config['fixtures']:
            name = fixture+('-warmup' if repetition == 0 else f'-r{repetition}')
            collect(args.endpoint, args.output, fixture, name, args.arm, runtime, warmup=repetition == 0, seed=args.seed)


if __name__ == '__main__':
    main()
