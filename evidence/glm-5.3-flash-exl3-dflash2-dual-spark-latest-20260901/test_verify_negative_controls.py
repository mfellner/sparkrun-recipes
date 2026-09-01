#!/usr/bin/env python3
"""Adversarial controls proving verify.py rejects forged evidence."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
RECIPE = ROOT.parents[1] / "recipes/glm-5.3-flash-exl3-dflash2-dual-spark-1m.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_hash(request: dict) -> str:
    return hashlib.sha256(json.dumps(request).encode()).hexdigest()


def load(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text())


def save(root: Path, rel: str, data: dict) -> None:
    (root / rel).write_text(json.dumps(data, indent=2) + "\n")


def update_manifest(root: Path, rels: list[str]) -> None:
    manifest = root / "SHA256SUMS"
    lines = manifest.read_text().splitlines()
    pending = set(rels)
    for index, line in enumerate(lines):
        for rel in tuple(pending):
            if line.endswith("  " + rel):
                lines[index] = f"{sha256(root / rel)}  {rel}"
                pending.remove(rel)
    if pending:
        raise RuntimeError(f"manifest paths missing: {sorted(pending)}")
    manifest.write_text("\n".join(lines) + "\n")


def run_case(name: str, mutate: Callable[[Path], list[str]], refresh_digest: bool = True) -> None:
    with tempfile.TemporaryDirectory(prefix="glm53-verify-negative-") as tmp:
        copy = Path(tmp) / "evidence"
        shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        changed = mutate(copy)
        if refresh_digest:
            update_manifest(copy, changed)
        env = os.environ.copy()
        env["GLM53_RECIPE"] = str(RECIPE)
        result = subprocess.run([sys.executable, str(copy / "verify.py")], text=True, capture_output=True, env=env, timeout=120)
        if result.returncode == 0:
            raise RuntimeError(f"negative control unexpectedly passed: {name}")
        print(f"{name}: rejected rc={result.returncode}")


def mutate_direct_content(root: Path) -> list[str]:
    rel = "acceptance.json"
    data = load(root, rel)
    data["checks"]["direct_exact"]["response"]["choices"][0]["message"]["content"] = "CORRUPTED"
    save(root, rel, data)
    return [rel]


def mutate_direct_controls(root: Path) -> list[str]:
    rel = "acceptance.json"
    data = load(root, rel)
    row = data["checks"]["direct_exact"]
    row["request"]["temperature"] = 0.75
    row["request_sha256"] = request_hash(row["request"])
    save(root, rel, data)
    return [rel]


def mutate_vision_fixture(root: Path) -> list[str]:
    rel = "acceptance.json"
    fixture_rel = "vision-quadrants.png"
    replacement = b"forged-not-the-red-green-blue-yellow-fixture"
    (root / fixture_rel).write_bytes(replacement)
    digest = hashlib.sha256(replacement).hexdigest()
    data = load(root, rel)
    data["fixture"]["sha256"] = digest
    data["fixture"]["size"] = len(replacement)
    url = "data:image/png;base64," + base64.b64encode(replacement).decode()
    for key in ("direct_vision", "proxy_vision"):
        row = data["checks"][key]
        row["request"]["messages"][0]["content"][1]["image_url"]["url"] = url
        row["request_sha256"] = request_hash(row["request"])
    save(root, rel, data)
    return [rel, fixture_rel]


def mutate_xgrammar(root: Path) -> list[str]:
    rel = "acceptance-xgrammar.json"
    data = load(root, rel)
    data["reasoning_c4"][0]["response"]["choices"][0]["message"]["content"] = '{"answer":42,"case":999}'
    save(root, rel, data)
    return [rel]


def mutate_apc_delta(root: Path) -> list[str]:
    rel = "acceptance-latest-features.json"
    data = load(root, rel)
    data["checks"]["apc"]["hit_delta"] += 1
    save(root, rel, data)
    return [rel]


def mutate_video_fixture(root: Path) -> list[str]:
    rel = "acceptance-latest-features.json"
    fixture_rel = "video-red-blue.gif"
    replacement = b"GIF89a-forged-not-red-then-blue"
    (root / fixture_rel).write_bytes(replacement)
    digest = hashlib.sha256(replacement).hexdigest()
    data = load(root, rel)
    data["checks"]["video"]["fixture_sha256"] = digest
    url = "data:image/gif;base64," + base64.b64encode(replacement).decode()
    for key in ("direct", "proxy"):
        row = data["checks"]["video"][key]
        row["request"]["messages"][0]["content"][1]["video_url"]["url"] = url
        row["request_sha256"] = request_hash(row["request"])
    save(root, rel, data)
    return [rel, fixture_rel]


def mutate_kpool(root: Path) -> list[str]:
    rel = "acceptance-latest-features.json"
    data = load(root, rel)
    data["checks"]["kpool_long_generation"]["result"]["response"]["usage"]["completion_tokens"] = 2299
    save(root, rel, data)
    return [rel]


def mutate_worker_nccl_output(root: Path) -> list[str]:
    rel = "raw/runtime-success.json"
    data = load(root, rel)
    data["hosts"]["192.168.178.46"]["nccl_runtime"]["stdout"] = "sha256=" + "0" * 64 + "\npackage_version=0.0.0\n"
    save(root, rel, data)
    return [rel]


def mutate_worker_nccl_command(root: Path) -> list[str]:
    rel = "raw/runtime-success.json"
    data = load(root, rel)
    data["hosts"]["192.168.178.46"]["nccl_runtime"]["argv"] = ["ssh", "192.168.178.46", "printf fabricated"]
    save(root, rel, data)
    return [rel]


def mutate_worker_nccl_row(root: Path) -> list[str]:
    rel = "raw/runtime-success.json"
    data = load(root, rel)
    data["hosts"]["192.168.178.46"]["nccl_runtime"] = copy.deepcopy(data["hosts"]["192.168.178.47"]["nccl_runtime"])
    save(root, rel, data)
    return [rel]


def mutate_worker_nccl_fake_path(root: Path) -> list[str]:
    rel = "raw/runtime-success.json"
    data = load(root, rel)
    row = data["hosts"]["192.168.178.46"]["nccl_runtime"]
    row["stdout"] = row["stdout"].replace(
        "path=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2",
        "path=/fake/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2",
    )
    save(root, rel, data)
    return [rel]


def mutate_e2_log_command(root: Path) -> list[str]:
    rel = "raw/runtime-success.json"
    data = load(root, rel)
    data["hosts"]["192.168.178.46"]["serve_log"]["argv"] = ["ssh", "192.168.178.46", "printf fabricated-e2-log"]
    save(root, rel, data)
    return [rel]


def mutate_runtime_time(root: Path) -> list[str]:
    rel = "raw/runtime-success.json"
    data = load(root, rel)
    data["captured_at"] = "2030-01-01T00:00:00+00:00"
    save(root, rel, data)
    return [rel]


def mutate_executor_security(root: Path) -> list[str]:
    rel = "raw/runtime-success.json"
    data = load(root, rel)
    inspect = json.loads(data["hosts"]["192.168.178.47"]["docker_inspect"]["stdout"])
    inspect[0]["HostConfig"]["Privileged"] = True
    data["hosts"]["192.168.178.47"]["docker_inspect"]["stdout"] = json.dumps(inspect)
    save(root, rel, data)
    return [rel]


def mutate_postready_kernel(root: Path) -> list[str]:
    rel = "raw/runtime-success.json"
    data = load(root, rel)
    data["hosts"]["192.168.178.47"]["kernel_log_after_acceptance"]["stdout"] += "injected NV_ERR_NO_MEMORY after final acceptance\n"
    save(root, rel, data)
    return [rel]


def mutate_public_manifest(root: Path) -> list[str]:
    rel = "raw/image-public.json"
    data = load(root, rel)
    data["anonymous_registry"]["token_url"] = "https://attacker.invalid/token"
    data["anonymous_registry"]["manifest_body_base64"] = base64.b64encode(b"{}").decode()
    data["anonymous_registry"]["manifest_body_sha256"] = "0" * 64
    data["anonymous_registry"]["manifest_size"] = 2
    save(root, rel, data)
    return [rel]


def mutate_delete_launch(root: Path) -> list[str]:
    rel = "raw/launch.log"
    (root / rel).unlink()
    manifest = root / "SHA256SUMS"
    manifest.write_text("\n".join(line for line in manifest.read_text().splitlines() if not line.endswith("  " + rel)) + "\n")
    return []


def main() -> int:
    run_case("manifest-integrity", mutate_direct_content, refresh_digest=False)
    run_case("direct-semantic-with-refreshed-manifest", mutate_direct_content)
    run_case("direct-controls-with-refreshed-manifest", mutate_direct_controls)
    run_case("vision-fixture-with-refreshed-manifest", mutate_vision_fixture)
    run_case("xgrammar-semantic-with-refreshed-manifest", mutate_xgrammar)
    run_case("apc-counter-with-refreshed-manifest", mutate_apc_delta)
    run_case("video-fixture-with-refreshed-manifest", mutate_video_fixture)
    run_case("kpool-outcome-with-refreshed-manifest", mutate_kpool)
    run_case("worker-nccl-output-with-refreshed-manifest", mutate_worker_nccl_output)
    run_case("worker-nccl-command-with-refreshed-manifest", mutate_worker_nccl_command)
    run_case("worker-nccl-row-swap-with-refreshed-manifest", mutate_worker_nccl_row)
    run_case("worker-nccl-fake-path-with-refreshed-manifest", mutate_worker_nccl_fake_path)
    run_case("e2-log-command-with-refreshed-manifest", mutate_e2_log_command)
    run_case("runtime-time-join-with-refreshed-manifest", mutate_runtime_time)
    run_case("executor-security-with-refreshed-manifest", mutate_executor_security)
    run_case("postready-kernel-with-refreshed-manifest", mutate_postready_kernel)
    run_case("public-manifest-with-refreshed-manifest", mutate_public_manifest)
    run_case("required-launch-receipt-deleted", mutate_delete_launch)
    print("all negative controls rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
