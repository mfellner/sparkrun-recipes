#!/usr/bin/env python3
"""Recompute the narrow per-rank runtime/safety claims from captured commands.

Verify pinned v1.7 run receipts; committed captures are historical, not live state.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
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
MODEL_SHA = "078455ffe6472f9a52fbc1139f58b9db2881b25c"
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
    created = [datetime.fromisoformat(receipt["hosts"][host]["created"].replace("Z", "+00:00")).timestamp()
               for host in HOSTS]
    require(runtime["kernel_since"] == int(min(created)) - 5 and
            runtime["kernel_since"] < min(created) < receipt["captured_at"] <
            receipt["completed_at"] < runtime["kernel_until"] <= runtime["captured_at"] + 2,
            "launch-to-acceptance kernel window")
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
    require(runtime["process_map_script_sha256"] == hashlib.sha256((HERE / "process_map.py").read_bytes()).hexdigest(), "process map source")
    pids = []
    for rank, host in enumerate(HOSTS):
        rows = runtime["hosts"][host]
        require(set(rows) == {"inspect", "processes", "process_map", "serve_log", "serve_hash",
                              "kernel", "kernel_repeat", "rdma", "earlyoom"}, "probe set")
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
        mapped = json.loads(command(rows["process_map"],
                                    ["ssh", host, "python3", "-", parts[0]]))
        require(mapped == {"host_pid": int(parts[0]), "namespace_pids": [int(parts[0]), mapped["namespace_pids"][1]],
                           "start_ticks": mapped["start_ticks"], "cmdline": serve} and
                mapped["namespace_pids"][1] > 1 and mapped["start_ticks"] > 0,
                "host-to-container PID/start mapping")
        if rank == 0:
            listener = json.loads(receipt["listener_before"]["stdout"])["owner"]
            require(listener["cmdline"] == serve and listener["pid"] == mapped["namespace_pids"][1] and
                    listener["start_ticks"] == mapped["start_ticks"],
                    "accepted port-8000 owner differs from serving process")
        for fragment in (f"--rank {rank}", "--tp 2", "--master 192.168.3.72", f"snapshots/{MODEL_SHA}",
                         f"snapshots/{DRAFT_SHA}", "--context 1048576", "--parallel 4", "--vision"):
            require(fragment in serve, f"serving argument {fragment}")
        require(("--port 8000" in serve and "--name " + MODEL in serve) == (rank == 0), "head API/rank")
        log = command(rows["serve_log"], ["ssh", host, "docker exec " + name +
            " python3 -c 'print(open(\"/tmp/sparkrun_serve.log\").read())'"])
        log_hash = command(rows["serve_hash"], ["ssh", host, "docker", "exec", name,
                                                  "sha256sum", "/tmp/sparkrun_serve.log"])
        require(log.endswith("\n") and log_hash ==
                hashlib.sha256(log[:-1].encode()).hexdigest() + "  /tmp/sparkrun_serve.log\n",
                "serve log bytes differ from independently hashed live file")
        require(log.startswith(f"TensorFold SparkRun rank={rank} fabric="), "missing launch log header")
        require("[tensorfold] building CUDA extension tensorfold_roce_v1" in log and
                "[tensorfold] building CUDA extension tensorfold_glm_exl3_v20" in log and
                f"[tensorfold] CUDA rank {rank} startup estimate" in log, "truncated startup log")
        require(("[tensorfold] serving " + MODEL + " at http://0.0.0.0:8000/v1" if rank == 0
                 else "[tensorfold] rank 1 ready in ") in log, "ready log")
        require(len(log) >= (3000 if rank == 0 else 1400), "truncated serve log")
        require(not re.search(r"Traceback|NCCL\s+(?:WARN|ERROR)|collective failed|CUDA error|out of memory|"
                              r"fatal error|NV_ERR_NO_MEMORY|device-side assert|SIGSEGV|Killed process", log, re.I),
                "fatal serve log")
        kernel = command(rows["kernel"], ["ssh", host, "journalctl", "-k", "--since",
            f"@{runtime['kernel_since']}", "--until", f"@{runtime['kernel_until']}",
            "--no-pager", "-o", "short-unix", "--show-cursor"])
        repeated = command(rows["kernel_repeat"], ["ssh", host, "journalctl", "-k", "--since",
            f"@{runtime['kernel_since']}", "--until", f"@{runtime['kernel_until']}",
            "--no-pager", "-o", "short-unix", "--show-cursor"])
        require(kernel == repeated and rows["kernel"]["finished_at"] <= rows["kernel_repeat"]["started_at"],
                "kernel query missing entries or reordered")
        lines = kernel.splitlines()
        require(len(lines) >= 2 and re.fullmatch(r"-- cursor: s=[A-Za-z0-9;=]+", lines[-1]) is not None,
                "missing journal coverage/cursor")
        entries = []
        for line in lines[:-1]:
            match = re.fullmatch(r"(\d+\.\d+) (dgx01|dgx02) kernel: (.+)", line)
            if match is None:
                raise AssertionError("unrecognized kernel output")
            require(match[2] == ("dgx01" if rank == 0 else "dgx02"), "kernel host mismatch")
            entries.append(float(match[1]))
        require(all(runtime["kernel_since"] <= stamp <= runtime["kernel_until"] for stamp in entries) and
                min(entries) < receipt["captured_at"], "kernel window missing launch interval")
        require(not re.search(r"NVRM.*Xid|NV_ERR_NO_MEMORY|out of memory|oom.kill|Killed process|"
                              r"NCCL ERROR|collective failed|GPU has fallen off", kernel, re.I),
                "kernel GPU/OOM error")
        rdma = command(rows["rdma"], ["ssh", host, "rdma", "link", "show"])
        links = re.findall(r"^link (\S+)/1 state (\S+) physical_state (\S+) netdev (\S+)\s*$",
                           rdma, re.M)
        require(len(links) == 4 and len({name for name, *_ in links}) == 4, "RDMA inventory")
        active = {name: (state, physical, netdev) for name, state, physical, netdev in links}
        require(active.get("rocep1s0f1") == ("ACTIVE", "LINK_UP", "enp1s0f1np1") and
                active.get("roceP2p1s0f1") == ("ACTIVE", "LINK_UP", "enP2p1s0f1np1"),
                "both selected CX-7 rails must be active")
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
