#!/usr/bin/env python3
"""Recompute the narrow per-rank runtime/safety claims from captured commands.

This is a historical receipt verifier, not a substitute for a fresh live probe.
"""
from __future__ import annotations

from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("glm53_tensorfold_verify", HERE / "verify.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
CLUSTER, IMAGE_ID, MODEL, verify = (_module.CLUSTER, _module.IMAGE_ID, _module.MODEL, _module.verify)

HOSTS = ("192.168.178.47", "192.168.178.46")
MODEL_SHA = "9eaebb7c4e96d983dcd538e18624622ba5b820a8"
DRAFT_SHA = "bf582e4eacc1810f76656d1811693ff6c6737d2a"

def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def command(row: dict, expected: list[str], *, allow_inactive: bool = False) -> str:
    require(row["argv"] == expected, f"command mismatch: {expected}")
    require(row["finished_at"] >= row["started_at"] > 0, "command interval")
    require(row["returncode"] == (3 if allow_inactive else 0), f"command failure: {expected}")
    require(not row["stderr"].strip(), f"command stderr: {expected}")
    return row["stdout"]


def verify_runtime(receipt: dict, runtime: dict) -> dict:
    accepted = verify(receipt)
    require(runtime["cluster_id"] == CLUSTER and runtime["recipe_sha256"] == receipt["recipe_sha256"], "recipe/cluster identity")
    require(runtime["acceptance_started_at"] == receipt["captured_at"] and
            runtime["acceptance_completed_at"] == receipt["completed_at"], "acceptance interval")
    require(runtime["acceptance_completed_at"] < runtime["captured_at"] < runtime["completed_at"], "runtime interval")
    all_commands = [runtime["proxy_status"], runtime["proxy_models"],
                    *(probe for records in runtime["hosts"].values() for probe in records.values())]
    require(all(runtime["captured_at"] <= row["started_at"] < row["finished_at"] <= runtime["completed_at"]
                for row in all_commands), "runtime probe outside capture interval")
    final_health = runtime["final_health"]
    require(final_health["url"] == "http://192.168.178.47:8000/health" and
            runtime["captured_at"] <= final_health["started_at"] < final_health["finished_at"] <= runtime["completed_at"] and
            max(row["finished_at"] for row in all_commands) <= final_health["started_at"],
            "final health interval/route")
    require(set(runtime["hosts"]) == set(HOSTS), "host set")
    pids = []
    for rank, host in enumerate(HOSTS):
        rows = runtime["hosts"][host]
        require(set(rows) == {"inspect", "processes", "serve_log", "kernel", "rdma", "earlyoom"}, "probe set")
        name = CLUSTER + f"_node_{rank}"
        inspect = json.loads(command(rows["inspect"], ["ssh", host, "docker", "inspect", name]))
        require(len(inspect) == 1, "container count")
        c = inspect[0]
        require(c["Id"] == receipt["hosts"][host]["id"] and c["Image"] == IMAGE_ID, "container identity")
        require(c["Name"] == "/" + name and c["State"]["Running"] is True and
                c["State"]["OOMKilled"] is False, "container state")
        require(c["Created"] == receipt["hosts"][host]["created"] and
                c["State"]["StartedAt"] == receipt["hosts"][host]["started_at"], "process restart")
        started = datetime.fromisoformat(c["State"]["StartedAt"].replace("Z", "+00:00")).timestamp()
        require(started < receipt["captured_at"], "workload started after acceptance")
        require(c["HostConfig"]["NetworkMode"] == c["HostConfig"]["IpcMode"] == "host" and
                c["HostConfig"]["Privileged"] is False, "security state")
        process = command(rows["processes"], ["ssh", host, "docker", "top", name, "-eo", "pid,ppid,cmd"])
        lines = [line for line in process.splitlines() if "/usr/local/bin/tensorfold serve " in line]
        require(len(lines) == 1, "one TensorFold serving process")
        parts = lines[0].split()
        require(parts[0].isdigit() and parts[1].isdigit(), "serving PID")
        pids.append(parts[0])
        serve = " ".join(parts[2:])
        for fragment in (f"--rank {rank}", "--tp 2", "--master 192.168.3.72", f"snapshots/{MODEL_SHA}",
                         f"snapshots/{DRAFT_SHA}", "--context 1048576", "--parallel 4", "--vision"):
            require(fragment in serve, f"serving argument {fragment}")
        require(("--port 8000" in serve and "--name " + MODEL in serve) == (rank == 0), "head API/rank")
        log = command(rows["serve_log"], ["ssh", host, "docker exec " + name +
            " python3 -c 'print(open(\"/tmp/sparkrun_serve.log\").read())'"])
        require(("[tensorfold] serving " + MODEL if rank == 0 else "rank 1 ready") in log, "ready log")
        require(not re.search(r"Traceback|NCCL WARN|CUDA error|out of memory|fatal error", log, re.I), "fatal serve log")
        kernel = command(rows["kernel"], ["ssh", host, "journalctl", "-k", "--since",
            f"@{int(receipt['captured_at'])}", "--until", "now", "--no-pager", "-o", "short-iso"])
        require(not re.search(r"NVRM.*Xid|NV_ERR_NO_MEMORY|out of memory|oom.kill|Killed process", kernel, re.I), "kernel GPU/OOM error")
        rdma = command(rows["rdma"], ["ssh", host, "rdma", "link", "show"])
        require(len(re.findall(r"state ACTIVE physical_state LINK_UP", rdma)) >= 1, "RDMA link")
        require(command(rows["earlyoom"], ["ssh", host, "systemctl", "is-active", "earlyoom"], allow_inactive=True).strip() == "inactive", "earlyoom state")
        require(rows["kernel"]["finished_at"] > receipt["completed_at"], "kernel capture before acceptance")
    proxy = json.loads(command(runtime["proxy_status"], ["sparkrun", "proxy", "status", "--json"]))
    require(proxy["running"] is True and proxy["port"] == 4000 and
            proxy["autodiscover"]["running"] is True, "proxy state")
    models = json.loads(command(runtime["proxy_models"], ["sparkrun", "proxy", "models", "--json"]))
    expected = {"model_name": MODEL, "api_base": "http://192.168.178.47:8000/v1"}
    require(models == [expected] and proxy["models"] == models, "proxy backend")
    health = runtime["final_health"]
    require(health["http"] == 200 and health["body"]["ok"] is True and
            health["body"]["backend"] == "tensorfold" and health["body"]["requests_running"] == 0, "final health")
    return {**accepted, "runtime_verified": True, "serving_pids": pids, "proxy_backend": expected["api_base"]}


if __name__ == "__main__":
    receipt = json.loads((HERE / "receipt.json").read_text())
    runtime = json.loads((HERE / "runtime.json").read_text())
    print(json.dumps(verify_runtime(receipt, runtime), indent=2))
