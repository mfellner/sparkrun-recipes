"""Foreground-only operator wrapper around unchanged capture/performance collectors.

Task artifact, not a service runtime. Each invocation owns and joins its telemetry
child and completes one fixture; it never schedules another arm or mutates Docker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

REPO = Path('/home/max/sparkrun-recipes')
ROOT = Path('/home/max/drama3-performance-20261014/foreground-completion')
HOST = '192.168.178.70'
sys.path.insert(0, str(REPO / 'scripts/drama3'))
import capture_runtime
import performance


def same_container(before, after):
    # Docker builds Mounts from an unordered map; retain all values and duplicates.
    left, right = dict(before), dict(after)
    for item in (left, right):
        item['mounts'] = sorted(item['mounts'], key=lambda value: json.dumps(value, sort_keys=True))
    return left == right

p = argparse.ArgumentParser()
p.add_argument('--arm', required=True)
p.add_argument('--fixture', required=True)
p.add_argument('--container', required=True)
p.add_argument('--image', required=True)
p.add_argument('--recipe', type=Path, required=True)
a = p.parse_args()
if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in a.arm + a.fixture):
    raise ValueError('unsafe task identifiers')
folder = ROOT / a.arm / a.fixture
folder.mkdir(parents=True, exist_ok=False)
config = json.loads(urllib.request.urlopen(f'http://{HOST}:8010/v1/runtime-profile', timeout=10).read())
runtime = capture_runtime.capture(a.container)
runtime['configured_profile'] = config
if runtime['container']['image_id'] != a.image or runtime['container']['user'] != '1000:1000':
    raise ValueError('image/user changed')
for name in ['FISH_DRAMA3_KERNEL_STATE','FISH_DRAMA3_KERNEL_FP8']:
    if config['kernel_options'][name.removeprefix('FISH_DRAMA3_KERNEL_').lower()] is not False:
        raise ValueError('unexpected numerical kernel')
if config['kernel_options']['tower_up'] != 'off' or config['kernel_options']['gdn_qkv'] != 'off':
    raise ValueError('unexpected quantization')
(folder / 'runtime-before.json').write_text(json.dumps(runtime, indent=2)+'\n')
(folder / 'harness.json').write_text(json.dumps(performance.harness_hashes(), indent=2)+'\n')
(folder / 'recipe.yaml').write_bytes(a.recipe.read_bytes())
remote_dir = f'/home/max/.cache/huggingface/drama3/performance/foreground-{a.arm}-{a.fixture}'
subprocess.run(['ssh', HOST, shlex.join(['mkdir','-p',remote_dir])], check=True)
# Exclusive path; reject any stale sampler/receipt rather than overwrite it.
subprocess.run(['ssh', HOST, shlex.join(['test','!','-e',remote_dir+'/samples.jsonl'])], check=True)
telemetry_args = ['ssh', HOST, shlex.join(['python3','-','--output',remote_dir+'/samples.jsonl','--stop-file',remote_dir+'/stop','--seconds','570'])]
receipt = {'started_at': time.time(), 'argv': sys.argv, 'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runtime_image_id': a.image, 'telemetry_argv': telemetry_args, 'telemetry_sha256': hashlib.sha256((REPO/'scripts/drama3/telemetry.py').read_bytes()).hexdigest(), 'fixture':a.fixture,'arm':a.arm,'completed':False}
telemetry_log = (folder/'telemetry-driver.log').open('w')
child = subprocess.Popen(telemetry_args, stdin=subprocess.PIPE, stdout=telemetry_log, stderr=subprocess.STDOUT)
child.stdin.write((REPO/'scripts/drama3/telemetry.py').read_bytes());child.stdin.close()
try:
    argv = ['/home/max/.hermes/cache/scratch/drama3-all-tests311/bin/python', str(REPO/'scripts/drama3/performance.py'), '--endpoint',f'http://{HOST}:8010','--output',str(folder/'repeated'),'--arm',a.arm,'--runtime-json',str(folder/'runtime-before.json'),'--fixtures',a.fixture,'--repeats','5','--seed','42']
    receipt['collector_argv'] = argv
    result = subprocess.run(argv, timeout=525)
    receipt['collector_rc'] = result.returncode
    if result.returncode:
        raise RuntimeError('collector failed')
    after = capture_runtime.capture(a.container)
    after['configured_profile'] = json.loads(urllib.request.urlopen(f'http://{HOST}:8010/v1/runtime-profile', timeout=10).read())
    (folder/'runtime-after.json').write_text(json.dumps(after,indent=2)+'\n')
    for key in ['container','process_join','runtime_files','configured_profile']:
        unchanged = same_container(runtime[key], after[key]) if key == 'container' else runtime[key] == after[key]
        if not unchanged:
            raise RuntimeError('runtime changed during collection: '+key)
    receipt['completed'] = True
finally:
    subprocess.run(['ssh', HOST, shlex.join(['touch',remote_dir+'/stop'])], check=True)
    try:
        receipt['telemetry_rc'] = child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.terminate();child.wait(timeout=10);receipt['telemetry_rc'] = child.returncode
    telemetry_log.close()
    copy = subprocess.run(['rsync','-a',HOST+':'+remote_dir+'/samples.jsonl',str(folder/'telemetry.jsonl')])
    receipt['telemetry_copy_rc'] = copy.returncode
    receipt['finished_at'] = time.time()
    (folder/'operator-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if receipt['telemetry_rc'] != 0 or receipt['telemetry_copy_rc'] != 0:
    raise RuntimeError('telemetry collection failed')
print(json.dumps({'foreground_batch':str(folder),'completed':True,'quality_qualified':False,'performance_qualified':False}),flush=True)
