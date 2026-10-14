#!/usr/bin/env python3
"""Fail closed on incomplete, mismatched, shortened or unbound performance arms. SPDX-License-Identifier: MIT."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from performance import audio_metrics, digest, first_pcm, identity
from fixtures import request_for
from acceptance import inspect_audio
from asr import MODEL as ASR_MODEL, REVISION as ASR_REVISION, word_error_rate, tail_matches


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_row(row, directory, runtime, harness):
    require(row.get('complete') is True and 'error' not in row, 'incomplete or failed request')
    require(row.get('status') == 200 and type(row.get('status')) is int, 'HTTP status')
    require(row.get('runtime') == runtime and row.get('runtime_identity') == identity(runtime), 'runtime substitution')
    require(runtime.get('container', {}).get('running') is True, 'runtime not running')
    require(runtime.get('health', {}).get('cudnn_enabled') is True, 'CUDA health boolean')
    require(runtime.get('health', {}).get('revision') == 'dd64a85cd82e9521f7202610ede97b367fc83946', 'runtime model revision')
    require(row.get('harness') == harness and bool(harness), 'harness substitution')
    fixture = row.get('fixture')
    try:
        expected = request_for(fixture)
    except (KeyError, TypeError) as exc:
        raise ValueError('unknown fixture') from exc
    request = row.get('request', {})
    # Non-default seeds are accepted only if explicitly recorded in the schedule.
    expected['seed'] = request.get('seed')
    require(type(expected['seed']) is int and 0 <= expected['seed'] <= 2147483647, 'seed')
    if fixture == 'reference':
        source = Path(directory)/'reference-input.wav'
        expected['references'] = [dict(audio='sha256:'+digest(source.read_bytes()), text=(Path(directory)/'reference.txt').read_text().strip(), consent=True)]
    require(request == expected and row.get('request_identity') == identity(expected), 'canonical request/sampling/reference mismatch')
    require(row.get('route') == '/v1/tts' and row.get('endpoint') == 'http://192.168.178.70:8010', 'route or endpoint mismatch')
    headers = row.get('headers', {})
    require(headers.get('x-model-revision') == 'dd64a85cd82e9521f7202610ede97b367fc83946', 'model header mismatch')
    require(headers.get('content-type', '').split(';')[0] == 'audio/wav', 'not audio WAV')
    require(re.fullmatch('[0-9a-f]{32}', row.get('request_id', '')) is not None and headers.get('x-request-id') == row['request_id'], 'request ID')
    filename = row.get('audio_file', '')
    require(filename == row.get('name', '')+'.wav' and Path(filename).name == filename, 'unsafe audio path')
    raw = (Path(directory)/filename).read_bytes()
    require(digest(raw) == row.get('audio_sha256'), 'audio digest')
    audio = inspect_audio(raw)
    require(audio == row.get('audio'), 'decoded sample/audio metadata mismatch')
    wall = row.get('wall_s', float('nan'))
    metrics = audio_metrics(audio['sample_rate'], audio['samples'], wall)
    require(all(type(row.get(k)) in (float, int) and math.isclose(row[k], v, rel_tol=1e-10, abs_tol=1e-10) for k, v in metrics.items()), 'RTF accounting')
    first = first_pcm(raw, row.get('events', []))
    require(row.get('first_pcm_s') == first and 0 < first <= wall and row['events'][-1]['elapsed_s'] <= wall, 'first PCM accounting')
    require(0 <= row.get('headers_s', -1) <= first, 'header ordering')
    start, end = row.get('started_at', 0), row.get('finished_at', 0)
    require(type(start) in (int, float) and type(end) in (int, float) and math.isfinite(start) and math.isfinite(end) and abs(end-start-wall) < 1, 'wall-clock correlation')
    require(type(row.get('warmup')) is bool, 'warmup must be boolean')
    return row


def spread(values):
    return dict(count=len(values), median=statistics.median(values), minimum=min(values), maximum=max(values), p95=sorted(values)[math.ceil(.95*len(values))-1])


def compare_rows(baseline, candidate):
    require(len(baseline) == len(candidate) and len(baseline) >= 5, 'matched repetitions required')
    groups = {(r['fixture'], r['request_identity']) for r in baseline}
    require(groups == {(r['fixture'], r['request_identity']) for r in candidate}, 'request/reference/sampling mismatch')
    fixture_results = {}
    for group in sorted(groups):
        left = [r for r in baseline if (r['fixture'], r['request_identity']) == group]
        right = [r for r in candidate if (r['fixture'], r['request_identity']) == group]
        require(len(left) == len(right) and len(left) >= 5, 'matched per-fixture repetitions required')
        ratio = statistics.median(r['audio']['samples'] for r in right)/statistics.median(r['audio']['samples'] for r in left)
        require(.9 <= ratio <= 1.1, 'audio length changed beyond predeclared 10% guard; quality review required')
        fixture_results[group[0]] = dict(baseline_rtf=spread([r['rtf'] for r in left]), candidate_rtf=spread([r['rtf'] for r in right]), median_sample_ratio=ratio)
    before, after = statistics.median(r['rtf'] for r in baseline), statistics.median(r['rtf'] for r in candidate)
    return dict(baseline_median_rtf=before, candidate_median_rtf=after, median_rtf_reduction=1-after/before, measured_per_arm=len(baseline), fixtures=fixture_results)


def aggregate(rows, fixtures, repeats=5):
    require(type(repeats) is int and repeats >= 5, 'missing repetitions')
    measured = [r for r in rows if r['warmup'] is False]
    require(set(r['fixture'] for r in rows) == set(fixtures), 'fixture set mismatch')
    require(all(sum(r['fixture'] == f for r in measured) == repeats for f in fixtures), 'missing repetitions')
    require(len({r['request_id'] for r in rows}) == len(rows), 'duplicate request ID')
    require(all(sum(r['fixture'] == f and r['warmup'] is True for r in rows) == 1 for f in fixtures), 'missing or duplicate warmup')
    return dict(measured_count=len(measured), warmup_count=len(rows)-len(measured), all_serial_rtf=spread([r['rtf'] for r in measured]), fixtures={f: dict(rtf=spread([r['rtf'] for r in measured if r['fixture'] == f]), first_pcm_s=spread([r['first_pcm_s'] for r in measured if r['fixture'] == f]), samples=spread([r['audio']['samples'] for r in measured if r['fixture'] == f])) for f in fixtures})


def validate_asr(row, result):
    expected = row['request']['text']
    text = result.get('text', '')
    wer = word_error_rate(expected, text)
    tail = tail_matches(expected, text)
    require(result.get('name') == row['name'] and result.get('passed') is True and result.get('tail_matches') is True and tail is True and result.get('wer') == wer and wer <= .4 and result.get('expected') == expected and result.get('audio_sha256') == row['audio_sha256'], 'ASR audio/text/ending mismatch')
    return True


def verify(directory, runtime, harness, require_asr=True):
    directory = Path(directory)
    config = json.loads((directory/'schedule.json').read_text())
    require(config['runtime'] == runtime and config['harness'] == harness, 'schedule provenance')
    require(config.get('warmups') == 1 and config.get('repeats', 0) >= 5, 'schedule repetitions')
    rows = []
    for rep in range(config['repeats']+1):
        for f in config['fixtures']:
            name = f+('-warmup' if rep == 0 else f'-r{rep}')
            row = json.loads((directory/(name+'.json')).read_text())
            require(row['name'] == name and row['arm'] == config['arm'] and row['request']['seed'] == config['seed'] and row['warmup'] is (rep == 0), 'schedule/row mismatch')
            rows.append(validate_row(row, directory, runtime, harness))
    if require_asr:
        asr = json.loads((directory/'asr.json').read_text())
        require(asr.get('passed') is True and asr.get('device') == 'cpu', 'ASR not passed')
        require(asr.get('model') == ASR_MODEL and asr.get('revision') == ASR_REVISION and asr.get('compute_type') == 'int8' and asr.get('beam_size') == 5 and asr.get('vad_filter') is False, 'ASR provenance/settings')
        require(len(asr.get('rows', [])) == len(rows), 'ASR completeness')
        require(len({r['name'] for r in asr['rows']}) == len(rows), 'duplicate ASR name')
        by_name = {r['name']: r for r in asr['rows']}
        for row in rows:
            validate_asr(row, by_name[row['name']])
    return aggregate(rows, config['fixtures'], config['repeats'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--without-asr', action='store_true', help='Diagnostic only, never a release qualification')
    args = parser.parse_args()
    result = verify(args.directory, json.loads(args.runtime.read_text()), json.loads(args.harness.read_text()), not args.without_asr)
    print(json.dumps(dict(qualified=not args.without_asr, **result), indent=2))


if __name__ == '__main__':
    main()
