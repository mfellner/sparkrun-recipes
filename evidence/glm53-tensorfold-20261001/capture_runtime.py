#!/usr/bin/env python3
"""Capture final per-rank process, serve, kernel and proxy receipts."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
CLUSTER = "sparkrun_98fa271a0dea8179_9e378c82c913"
HOSTS = ("192.168.178.47", "192.168.178.46")
receipt = json.loads((HERE / "receipt.json").read_text())


def cmd(*args: str) -> dict:
    started = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    return {"argv": list(args), "started_at": started, "finished_at": time.time(),
            "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


result = {"cluster_id": CLUSTER, "recipe_sha256": receipt["recipe_sha256"],
          "acceptance_started_at": receipt["captured_at"], "acceptance_completed_at": receipt["completed_at"],
          "captured_at": time.time(), "hosts": {}}
for rank, host in enumerate(HOSTS):
    container = CLUSTER + f"_node_{rank}"
    records = {}
    records["inspect"] = cmd("ssh", host, "docker", "inspect", container)
    records["processes"] = cmd("ssh", host, "docker", "top", container, "-eo", "pid,ppid,cmd")
    records["serve_log"] = cmd("ssh", host, shlex.join(["docker", "exec", container, "python3", "-c",
                               'print(open("/tmp/sparkrun_serve.log").read())']))
    records["kernel"] = cmd("ssh", host, "journalctl", "-k", "--since", f"@{int(receipt['captured_at'])}",
                            "--until", "now", "--no-pager", "-o", "short-iso")
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
