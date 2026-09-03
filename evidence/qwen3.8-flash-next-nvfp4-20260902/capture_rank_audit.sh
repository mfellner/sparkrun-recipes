#!/usr/bin/env bash
set -euo pipefail
container="${1:?container name required}"
printf 'host='; hostname
printf 'captured='; date --iso-8601=seconds
docker inspect "$container" | python3 -c 'import json,sys; d=json.load(sys.stdin)[0]; st=d.get("State",{}); hc=d.get("HostConfig",{}); h=st.get("Health",{}).get("Status","none"); print("container={} id={} running={} status={} image={} network={} ipc={} privileged={} user={} health={}".format(d.get("Name"),d.get("Id"),st.get("Running"),st.get("Status"),d.get("Image"),hc.get("NetworkMode"),hc.get("IpcMode"),hc.get("Privileged"),d.get("Config",{}).get("User"),h))'
printf '\nenvironment\n'
docker inspect "$container" --format '{{json .Config.Env}}' | python3 -c 'import json,sys; wanted=("NCCL_","GLOO_","VLLM_","HF_HOME","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","PLE_QUANT_OVERRIDE","QWEN38_"); print("\n".join(sorted(x for x in json.load(sys.stdin) if x.startswith(wanted))))'
printf '\nprocesses\n'
docker top "$container" -eo pid,ppid,stat,etime,pcpu,pmem,comm,args
printf '\nselected_serve_log\n'
docker exec "$container" python3 -c 'from pathlib import Path; s=Path("/tmp/sparkrun_serve.log").read_text(errors="replace").splitlines(); keys=("Init COMPLETE","Connected all rings","NET/IB","Model loading took","Available KV cache memory","GPU KV cache size","Maximum concurrency","Application startup complete","Uvicorn running","POST /v1/chat/completions","GET /v1/models","Traceback","OutOfMemory","NCCL WARN","NCCL error","CUDA error","EngineCore encountered"); print("\n".join(line for line in s if any(key.lower() in line.lower() for key in keys)))'
printf '\nselected_kernel_log\n'
started_at=$(docker inspect "$container" --format '{{.State.StartedAt}}')
journalctl -k -b --since "$started_at" --no-pager | python3 -c 'import sys; keys=("xid","nvrm","oom-kill","killed process","out of memory"); print("".join(line for line in sys.stdin if any(key in line.lower() for key in keys)),end="")'
printf '\ngpu\n'
nvidia-smi --query-gpu=name,pstate,temperature.gpu,utilization.gpu,clocks.sm,power.draw --format=csv,noheader
