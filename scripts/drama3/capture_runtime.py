#!/usr/bin/env python3
"""Capture exact scoped target/container and listener joins. SPDX-License-Identifier: MIT."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

HOST = "192.168.178.70"
FILES = ["contract.py", "server.py", "run.sh", "probe.py", "model-manifest.json", "requirements-lock.txt", "default-reference.wav", "default-reference.txt", "reference-provenance.json", "manifest.sha256"]


def remote(args, input=None):
    return subprocess.check_output(["ssh", HOST, *args], input=input, text=True, timeout=120)


def capture(container):
    inspect = json.loads(remote(["docker", "inspect", container]))[0]
    config, state, host = inspect["Config"], inspect["State"], inspect["HostConfig"]
    allowed_env = {"TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "HF_HOME", "FISH_DRAMA3_CUDA_GRAPH", "FISH_DRAMA3_FP8_EXPERTS", "DRAMA_MODEL_DIR", "DRAMA_OUTPUT_DIR", "OMP_NUM_THREADS"}
    effective_env = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in config["Env"] if item.split("=", 1)[0] in allowed_env}
    health = json.load(urllib.request.urlopen(f"http://{HOST}:8010/health", timeout=10))
    # Do not serialize raw Docker environment/config (can contain credentials).
    code = '''import pathlib,json,hashlib,socket,os
pid=int(os.environ["CAPTURE_PID"])
entries=[]
for file in ("/proc/net/tcp","/proc/net/tcp6"):
 for line in pathlib.Path(file).read_text().splitlines()[1:]:
  row=line.split()
  if row[3]=="0A" and int(row[1].split(":")[1],16)==8010:
   entries.append(row)
assert len(entries)==1, "8010 has extra/wildcard/IPv6 listeners"
row=entries[0]
address=socket.inet_ntoa(bytes.fromhex(row[1].split(":")[0])[::-1])
assert address=="192.168.178.70"
inode=row[9]
assert any(os.readlink(fd)=="socket:["+inode+"]" for fd in pathlib.Path(f"/proc/{pid}/fd").iterdir()), "API PID does not own listener"
stat=pathlib.Path(f"/proc/{pid}/stat").read_text().split(")",1)[1].split()
files={name:hashlib.sha256((pathlib.Path("/opt/drama")/name).read_bytes()).hexdigest() for name in ''' + repr(FILES) + '''}
print(json.dumps(dict(listener=dict(address="192.168.178.70:8010",pid=pid,start_ticks=int(stat[19]),inode=inode),runtime_files=files)))'''
    import shlex
    result = json.loads(remote(["docker", "exec", "-e", "CAPTURE_PID=" + str(health["pid"]), container, "python", "-c", shlex.quote(code)]))
    pids = remote(["docker", "top", container, "-eo", "pid"]).splitlines()[1:]
    join = None
    for pid in pids:
        pid = pid.strip()
        status = remote(["cat", f"/proc/{pid}/status"])
        ns = next((line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:")), [])
        if ns and int(ns[-1]) == health["pid"]:
            stat = remote(["cat", f"/proc/{pid}/stat"]).split(")", 1)[1].split()
            join = dict(host_pid=int(pid), namespace_pids=[int(value) for value in ns], host_start_ticks=int(stat[19]))
            break
    if not join or join["host_start_ticks"] != result["listener"]["start_ticks"]:
        raise ValueError("host/container process start identity join failed")
    return dict(host=HOST, hostname=remote(["hostname"]).strip(), captured_at=time.time(), effective_env=effective_env, container=dict(id=inspect["Id"], image_ref=config["Image"], image_id=inspect["Image"], running=state["Running"], created_at=datetime.fromisoformat(inspect["Created"].replace("Z", "+00:00")).timestamp(), started_at=state["StartedAt"], privileged=host["Privileged"], user=config["User"], network=host["NetworkMode"], ipc=host["IpcMode"], security_opt=host["SecurityOpt"], cap_add=host["CapAdd"], devices=host["Devices"], mounts=[dict(source=item["Source"], destination=item["Destination"], rw=item["RW"]) for item in inspect["Mounts"]]), health=health, process_join=join, **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(capture(args.container), indent=2) + "\n")
    print(args.output)
