#!/usr/bin/env python3
"""Capture final per-rank process, serve, kernel and proxy receipts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
PROCESS_SCRIPT = HERE / "process_map.py"
CLUSTER = "sparkrun_ce1b4db30465bce3_85f608d441ab"
HOSTS = ("192.168.178.47", "192.168.178.46")
receipt = json.loads((HERE / "receipt.json").read_text())
created = [datetime.fromisoformat(receipt["hosts"][host]["created"].replace("Z", "+00:00")).timestamp()
           for host in HOSTS]
KERNEL_SINCE = int(min(created)) - 5
KERNEL_UNTIL = int(time.time()) + 1
assert KERNEL_SINCE < min(created) < receipt["captured_at"] < receipt["completed_at"] < KERNEL_UNTIL


def cmd(*args: str) -> dict:
    started = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    return {"argv": list(args), "started_at": started, "finished_at": time.time(),
            "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def cmd_stdin(source: str, *args: str) -> dict:
    started = time.time()
    proc = subprocess.run(args, input=source, capture_output=True, text=True, timeout=120)
    return {"argv": list(args), "started_at": started, "finished_at": time.time(),
            "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


result = {"cluster_id": CLUSTER, "recipe_sha256": receipt["recipe_sha256"],
          "process_map_script_sha256": hashlib.sha256(PROCESS_SCRIPT.read_bytes()).hexdigest(),
          "acceptance_started_at": receipt["captured_at"], "acceptance_completed_at": receipt["completed_at"],
          "kernel_since": KERNEL_SINCE, "kernel_until": KERNEL_UNTIL,
          "captured_at": time.time(), "hosts": {}}
for rank, host in enumerate(HOSTS):
    container = CLUSTER + f"_node_{rank}"
    records = {}
    records["inspect"] = cmd("ssh", host, "docker", "inspect", container)
    records["processes"] = cmd("ssh", host, "docker", "top", container, "-eo", "pid,ppid,cmd")
    serving = [line for line in records["processes"]["stdout"].splitlines()
               if "/usr/local/bin/tensorfold serve " in line]
    assert len(serving) == 1 and serving[0].split()[0].isdigit()
    records["process_map"] = cmd_stdin(PROCESS_SCRIPT.read_text(), "ssh", host,
                                       "python3", "-", serving[0].split()[0])
    records["serve_log"] = cmd("ssh", host, shlex.join(["docker", "exec", container, "python3", "-c",
                               'print(open("/tmp/sparkrun_serve.log").read())']))
    records["serve_hash"] = cmd("ssh", host, "docker", "exec", container,
                                 "sha256sum", "/tmp/sparkrun_serve.log")
    records["kernel"] = cmd("ssh", host, "journalctl", "-k", "--since", f"@{KERNEL_SINCE}",
                            "--until", f"@{KERNEL_UNTIL}", "--no-pager", "-o", "short-unix", "--show-cursor")
    records["kernel_repeat"] = cmd("ssh", host, "journalctl", "-k", "--since", f"@{KERNEL_SINCE}",
                                   "--until", f"@{KERNEL_UNTIL}", "--no-pager", "-o", "short-unix", "--show-cursor")
    records["rdma"] = cmd("ssh", host, "rdma", "link", "show")
    records["earlyoom"] = cmd("ssh", host, "systemctl", "is-active", "earlyoom")
    result["hosts"][host] = records
result["proxy_status"] = cmd("sparkrun", "proxy", "status", "--json")
result["proxy_models"] = cmd("sparkrun", "proxy", "models", "--json")
health_url = "http://192.168.178.47:8000/health"
health_started = time.time()
with urllib.request.urlopen(health_url, timeout=20) as response:
    result["final_health"] = {"url": health_url, "http": response.status, "body": json.loads(response.read())}
result["final_health"].update(started_at=health_started, finished_at=time.time())
result["completed_at"] = time.time()
out = HERE / "runtime.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(f"Wrote {out} ({out.stat().st_size} bytes)")
