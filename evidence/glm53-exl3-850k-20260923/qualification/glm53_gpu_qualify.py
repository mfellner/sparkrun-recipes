import concurrent.futures
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import yaml
ROOT = Path('/home/max/sparkrun-recipes')
SCRATCH = Path('/home/max/.hermes/cache/scratch')
SNAPSHOT = SCRATCH / 'glm53-final-gpu-qualification-20260923'
recipe = yaml.safe_load((ROOT / 'recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml').read_text())
shutil.copytree(ROOT / 'mods/glm-5.3-flash-exl3-upstream-850k', SNAPSHOT, ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
subprocess.run(['scp','-rq',str(SNAPSHOT),'192.168.178.46:'+str(SCRATCH)+'/'],check=True)
def probe(host):
    argv=['docker','run','--rm','--network','host','--memory','16g','--memory-swap','16g','--device','nvidia.com/gpu=all','--entrypoint','bash','-v',str(SNAPSHOT)+':/review/mod:ro']
    for key,value in recipe['env'].items():
        argv+=['-e',f'{key}={value}']
    argv+=['-e','TMPDIR=/review/tmp','-e','NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1','-e','NCCL_IB_GID_INDEX=3','-e','EXL3_SELFCHECK_GPU=1',recipe['container'],'-c','set -euo pipefail; mkdir -p /review/tmp; cd /review/mod; bash run.sh; python3 run_upstream_gpu_compatibility.py; sha256sum /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2']
    started=time.time()
    result=subprocess.run(['ssh',host,shlex.join(argv)],text=True,capture_output=True,timeout=540)
    record={'host':host,'started_at':started,'completed_at':time.time(),'argv':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    path=SCRATCH / ('glm53-final-gpu-'+host+'-20260923.json')
    path.write_text(json.dumps(record,indent=2))
    print(json.dumps({'host':host,'returncode':result.returncode,'receipt':str(path),'markers':[line for line in result.stdout.splitlines() if 'verify OK' in line or 'OFF64_PASS' in line or 'grouped OK' in line]}),flush=True)
    return result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    codes=list(pool.map(probe,['192.168.178.47','192.168.178.46']))
raise SystemExit(int(any(codes)))
