#!/usr/bin/env python3
"""Capture raw, command-bound runtime receipts from the gx10 service."""

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

DEFAULT_HOST = "192.168.178.51"
DEFAULT_CONTAINER = "sparkrun_7d7b52c2c9174082_15b33f409193_solo"
IMAGE = "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5@sha256:f92b4a1a476fd1e235e97df2a623c04c24b69d607a78a91f5084627ec7bd4266"
SNAPSHOT = "/cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-8B/snapshots/2c4565515e0f265c6511776e7193b22c0968ddc7"


def capture_remote(host, name, args):
    remote = shlex.join(args)
    command = ["ssh", "-o", "BatchMode=yes", host, remote]
    started_unix_ns = time.time_ns()
    completed = subprocess.run(command, text=True, capture_output=True, timeout=180, check=False)
    finished_unix_ns = time.time_ns()
    return {
        "name": name,
        "command": shlex.join(command),
        "started_unix_ns": started_unix_ns,
        "finished_unix_ns": finished_unix_ns,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def http_get(url):
    started_unix_ns = time.time_ns()
    with urllib.request.urlopen(url, timeout=15) as response:
        body = response.read().decode(errors="replace")
        status = response.status
    return {
        "url": url,
        "started_unix_ns": started_unix_ns,
        "finished_unix_ns": time.time_ns(),
        "status": status,
        "body": body,
    }


def receipt_ok(receipt):
    if receipt["returncode"] == 0:
        return True
    return (
        receipt["name"] == "kernel_fatal_scan"
        and receipt["returncode"] == 1
        and receipt["stdout"].strip() == "-- No entries --"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--acceptance-result", default=str(pathlib.Path(__file__).with_name("acceptance-result.json")))
    parser.add_argument("--non-regression-result", default=str(pathlib.Path(__file__).with_name("non-regression-result.json")))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    acceptance_path = pathlib.Path(args.acceptance_result)
    nonreg_path = pathlib.Path(args.non_regression_result)
    acceptance = json.loads(acceptance_path.read_text())
    nonreg = json.loads(nonreg_path.read_text())
    acceptance_started = acceptance["timestamp_utc"]

    model_check = (
        "import json,pathlib; "
        f"p=pathlib.Path({SNAPSHOT!r}); "
        "idx=json.loads((p/'model.safetensors.index.json').read_text()); "
        "shards=sorted(set(idx['weight_map'].values())); "
        "print(json.dumps({'snapshot':str(p),'exists':p.exists(),'shards':shards,"
        "'missing':[x for x in shards if not (p/x).exists()],"
        "'broken_symlinks':[x.name for x in p.iterdir() if x.is_symlink() and not x.exists()],"
        "'shard_bytes':sum((p/x).stat().st_size for x in shards)},sort_keys=True))"
    )
    log_read = (
        "import hashlib,json,pathlib; p=pathlib.Path('/tmp/sparkrun_serve.log'); b=p.read_bytes(); "
        "print(json.dumps({'path':str(p),'bytes':len(b),'mtime_ns':p.stat().st_mtime_ns,"
        "'sha256':hashlib.sha256(b).hexdigest(),'content':b.decode(errors='replace')},sort_keys=True))"
    )
    runtime_versions = (
        "import json,torch,vllm; "
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'vllm':vllm.__version__,"
        "'nccl':'.'.join(map(str,torch.cuda.nccl.version()))},sort_keys=True))"
    )

    receipts = [
        capture_remote(args.host, "host_identity", ["hostname"]),
        capture_remote(args.host, "container_inspect", ["docker", "inspect", args.container]),
        capture_remote(args.host, "image_inspect", ["docker", "image", "inspect", IMAGE]),
        capture_remote(args.host, "runtime_versions", ["docker", "exec", args.container, "python3", "-c", runtime_versions]),
        capture_remote(args.host, "model_snapshot", ["docker", "exec", args.container, "python3", "-c", model_check]),
        capture_remote(
            args.host,
            "cache_identity_and_writability",
            [
                "docker",
                "exec",
                args.container,
                "sh",
                "-lc",
                "id; stat -c '%U:%G %u:%g %a %n' /cache/huggingface /cache/huggingface/sparkrun-runtime-cache/qwen3-vl-embedding-8b /cache/huggingface/sparkrun-runtime-cache/qwen3-vl-embedding-8b/vllm; test -w /cache/huggingface/sparkrun-runtime-cache/qwen3-vl-embedding-8b/vllm",
            ],
        ),
        capture_remote(args.host, "container_processes", ["docker", "top", args.container, "-eo", "pid,ppid,user,stat,etime,cmd"]),
        capture_remote(
            args.host,
            "gpu_processes",
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        ),
        capture_remote(args.host, "host_memory", ["free", "-h"]),
        capture_remote(args.host, "serve_log", ["docker", "exec", args.container, "python3", "-c", log_read]),
        capture_remote(
            args.host,
            "kernel_fatal_scan",
            ["journalctl", "-k", "-b", "--since", acceptance_started, "--no-pager", "-g", "NVRM|Xid|oom-kill"],
        ),
    ]

    recipe = pathlib.Path(__file__).parents[2] / "recipes" / "qwen3-vl-embedding-8b-dgx-spark.yaml"
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "container": args.container,
        "acceptance_run_id": acceptance["run_id"],
        "acceptance_result_sha256": hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
        "acceptance_started_utc": acceptance["timestamp_utc"],
        "acceptance_finished_utc": acceptance["finished_utc"],
        "non_regression_run_id": nonreg.get("run_id"),
        "non_regression_result_sha256": hashlib.sha256(nonreg_path.read_bytes()).hexdigest(),
        "non_regression_timestamp_utc": nonreg["timestamp_utc"],
        "non_regression_finished_utc": nonreg["finished_utc"],
        "recipe": str(recipe),
        "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
        "http": [
            http_get(f"http://{args.host}:8000/health"),
            http_get(f"http://{args.host}:8000/v1/models"),
        ],
        "receipts": receipts,
    }
    all_commands_ok = all(receipt_ok(receipt) for receipt in receipts)
    output = pathlib.Path(args.output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "receipt_count": len(receipts), "all_commands_ok": all_commands_ok}, indent=2))
    raise SystemExit(0 if all_commands_ok else 1)


if __name__ == "__main__":
    main()
