#!/usr/bin/env python3
"""Capture exact image and checkpoint integrity on both live rank hosts.

No credentials are sent or recorded. The signed-by-content pinned HF manifest and
all snapshot bytes are checked independently on the local head and CX-7 worker.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts/verify_tensorfold_v171_snapshot.py"
MANIFEST = HERE / "hf-manifest.json"
REV = "078455ffe6472f9a52fbc1139f58b9db2881b25c"
MODEL_DIR = "/home/max/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold"
IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:a8067cd7e14c14fa83d1dbed60261428f6d1737cec4554445573354af040dd7c"
HOSTS = ("192.168.178.47", "192.168.178.46")


def probe(argv: list[str], stdin: str | None = None) -> dict:
    started = time.time()
    result = subprocess.run(argv, input=stdin, text=True, capture_output=True, timeout=900)
    return {"argv": argv, "started_at": started, "finished_at": time.time(),
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def on_host(host: str) -> dict:
    image = probe(["ssh", "-o", "BatchMode=yes", host, "docker", "image", "inspect", IMAGE])
    snapshot = probe(["ssh", "-o", "BatchMode=yes", host, "python3", "-",
                      "--snapshot", f"{MODEL_DIR}/snapshots/{REV}",
                      "--manifest", "/home/max/.hermes/cache/scratch/tf-v171-hf-manifest.json"], SCRIPT.read_text())
    return {"image": image, "snapshot": snapshot}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        items = list(pool.map(on_host, HOSTS))
    receipt = {"schema": 1, "captured_at": time.time(),
               "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
               "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
               "hosts": dict(zip(HOSTS, items, strict=True))}
    path = HERE / "staging.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    assert all(row[part]["returncode"] == 0 for row in items for part in ("image", "snapshot")), "staging probe failed"
    print(f"Captured {path} ({path.stat().st_size} bytes)")
