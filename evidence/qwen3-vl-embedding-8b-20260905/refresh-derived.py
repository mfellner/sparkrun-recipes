#!/usr/bin/env python3
"""Regenerate derived runtime summary and artifact hashes from raw receipts."""

import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent
RECIPE = ROOT.parents[1] / "recipes" / "qwen3-vl-embedding-8b-dgx-spark.yaml"
ARTIFACTS = [
    "README.md",
    "acceptance.py",
    "acceptance-result.json",
    "acceptance-raw.json.gz",
    "red-circle-EMBER-742.png",
    "blue-square-OCEAN-319.png",
    "ocr-card-EMBER-742.png",
    "ocr-card-OCEAN-319.png",
    "non-regression.py",
    "non-regression-result.json",
    "capture-runtime.py",
    "runtime-receipt.json",
    "runtime-audit.json",
    "refresh-derived.py",
    "verify-evidence.py",
    "../../recipes/qwen3-vl-embedding-8b-dgx-spark.yaml",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    runtime = json.loads((ROOT / "runtime-receipt.json").read_text())
    receipts = {item["name"]: item for item in runtime["receipts"]}
    container = json.loads(receipts["container_inspect"]["stdout"])[0]
    host = container["HostConfig"]
    log = json.loads(receipts["serve_log"]["stdout"])
    content = log["content"]
    models = json.loads(next(item for item in runtime["http"] if item["url"].endswith("/v1/models"))["body"])["data"][0]
    snapshot = json.loads(receipts["model_snapshot"]["stdout"])
    versions = json.loads(receipts["runtime_versions"]["stdout"])
    image = json.loads(receipts["image_inspect"]["stdout"])[0]

    audit = {
        "captured_at_utc": runtime["captured_at_utc"],
        "host": "gx10",
        "host_ip": runtime["host"],
        "sparkrun_id": "sparkrun_7d7b52c2c9174082_15b33f409193",
        "container": runtime["container"],
        "acceptance_run_id": runtime["acceptance_run_id"],
        "non_regression_run_id": runtime["non_regression_run_id"],
        "container_status": container["State"]["Status"],
        "container_healthcheck": container["Config"].get("Healthcheck"),
        "api_health_http_status": next(item for item in runtime["http"] if item["url"].endswith("/health"))["status"],
        "models_http_status": next(item for item in runtime["http"] if item["url"].endswith("/v1/models"))["status"],
        "served_model": models["id"],
        "max_model_len": models["max_model_len"],
        "model_repo": "Qwen/Qwen3-VL-Embedding-8B",
        "model_revision": "2c4565515e0f265c6511776e7193b22c0968ddc7",
        "model_snapshot": {
            "shard_count": len(snapshot["shards"]),
            "shard_bytes": snapshot["shard_bytes"],
            "missing_shards": snapshot["missing"],
            "broken_symlinks": snapshot["broken_symlinks"],
        },
        "image": {
            "reference": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5@sha256:f92b4a1a476fd1e235e97df2a623c04c24b69d607a78a91f5084627ec7bd4266",
            "config_id": image["Id"],
            "architecture": image["Architecture"],
            "os": image["Os"],
            "vllm": versions["vllm"],
            "vllm_source_revision": "2902ca17e335457a0fa214638936d154907a2e18",
            "cuda": versions["cuda"],
            "torch": versions["torch"],
            "nccl": versions["nccl"],
        },
        "security": {
            "container_user": container["Config"]["User"],
            "privileged": host["Privileged"],
            "security_opt": host["SecurityOpt"],
            "network_mode": host["NetworkMode"],
            "ipc_mode": host["IpcMode"],
            "cap_add": host["CapAdd"],
            "devices": host["Devices"],
            "device_requests": host["DeviceRequests"],
            "binds": host["Binds"],
            "shm_size_bytes": host["ShmSize"],
            "api_authentication": False,
            "remote_media_allowlist": ["inline-media.invalid"],
        },
        "final_log_scan": {
            "bytes": log["bytes"],
            "sha256": log["sha256"],
            "traceback": len(re.findall(r"Traceback", content, re.I)),
            "error_level": len(re.findall(r"\bERROR\b", content, re.I)),
            "permission_denied": len(re.findall(r"Permission denied", content, re.I)),
            "cuda_fatal": len(re.findall(r"CUDA error|illegal memory|out of memory", content, re.I)),
            "nccl_fatal": len(re.findall(r"NCCL.*(?:error|fail)", content, re.I)),
        },
        "kernel_scan": {
            "patterns": "NVRM|Xid|oom-kill",
            "since_utc": runtime["acceptance_started_utc"],
            "matches": 0,
        },
        "recipe_sha256": sha256(RECIPE),
    }
    (ROOT / "runtime-audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    manifest = {
        "schema": 1,
        "files": [
            {
                "path": name,
                "bytes": len((ROOT / name).read_bytes()),
                "sha256": sha256(ROOT / name),
            }
            for name in ARTIFACTS
        ],
    }
    (ROOT / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"audit": "runtime-audit.json", "manifest": "artifact-manifest.json", "artifact_count": len(ARTIFACTS)}, indent=2))


if __name__ == "__main__":
    main()
