#!/usr/bin/env python3
"""Preserve SparkRun's launch-persisted recipe state and log, without secrets."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import yaml

HERE = Path(__file__).resolve().parent
CLUSTER = "sparkrun_ce1b4db30465bce3_85f608d441ab"
JOB = Path.home() / ".cache/sparkrun/jobs/ce1b4db30465bce3_85f608d441ab.yaml"
LOG = Path.home() / ".hermes/cache/scratch/tf-v171-launch.log"
RECIPE = HERE.parents[1] / "recipes/glm-5.3-flash-exl3-tensorfold-dual-spark-1m-v171.yaml"
HOSTS = ("192.168.178.47", "192.168.178.46")
MOD_FILES = ("SHA256SUMS", "serve.sh", "run.sh", "fabric.py")


def probe(*argv: str) -> dict:
    start = time.time()
    p = subprocess.run(argv, text=True, capture_output=True, timeout=80)
    return {"argv": list(argv), "started_at": start, "finished_at": time.time(),
            "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


raw = JOB.read_bytes()
meta = yaml.safe_load(raw)
assert meta["cluster_id"] == CLUSTER and not meta.get("api_key")
assert meta["recipe_state"]["_raw"] == yaml.safe_load(RECIPE.read_text())
assert meta["recipe_state"]["_applied_overrides"] == {}
assert all(not key.lower().endswith(("password", "secret", "api_key", "token"))
           for key in meta["recipe_state"]["_raw"]["env"])
log = LOG.read_text()
assert f"Cluster:   {CLUSTER}" in log and "Container image up-to-date on all 2 host(s)" in log
record = {"cluster_id": CLUSTER, "captured_at": time.time(),
          "job_path": str(JOB), "job_mtime_ns": JOB.stat().st_mtime_ns,
          "job_ctime_ns": JOB.stat().st_ctime_ns, "job_sha256": hashlib.sha256(raw).hexdigest(),
          "job_yaml": raw.decode(), "recipe_sha256": hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
          "launch_log_path": str(LOG), "launch_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
          "launch_log": log, "hosts": {}}
for rank, host in enumerate(HOSTS):
    container = CLUSTER + f"_node_{rank}"
    hashes = [probe("ssh", "-o", "BatchMode=yes", host, "docker", "exec", container,
                    "sha256sum", *(f"/workspace/mods/glm53-tensorfold-1m/{p}" for p in MOD_FILES))]
    inspect = probe("ssh", "-o", "BatchMode=yes", host, "docker", "inspect", container)
    assert all(x["returncode"] == 0 for x in (*hashes, inspect))
    record["hosts"][host] = {"inspect": inspect, "mod_sha256": hashes[0]}
assert record["job_mtime_ns"] <= min(record["hosts"][h]["inspect"]["started_at"] for h in HOSTS) * 10**9
record["completed_at"] = time.time()
path = HERE / "launch.json"
path.write_text(json.dumps(record, indent=2) + "\n")
print(f"Captured {path} ({path.stat().st_size} bytes)")
