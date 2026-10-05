#!/usr/bin/env python3
"""Verify every byte in the pinned TensorFold HF snapshot, offline.

The manifest contains immutable-revision Git blob IDs and LFS SHA-256 digests
from the Hub's expanded pinned-revision API. Run this on each serving host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence/glm53-tensorfold-20261005/hf-manifest.json"
REVISION = "078455ffe6472f9a52fbc1139f58b9db2881b25c"
REPO = "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold"
DEFAULT_SNAPSHOT = (Path.home() / ".cache/huggingface/hub" /
                    "models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold" /
                    "snapshots" / REVISION)


def verify(snapshot: Path, manifest: dict) -> dict:
    assert manifest["id"] == REPO and manifest["sha"] == REVISION, "wrong Hub revision"
    files = manifest["files"]
    assert len(files) == len(set(files)) >= 90, "incomplete metadata"
    assert len([n for n in files if n.startswith("model-") and n.endswith(".safetensors")]) == 83, "shards"
    assert {str(p.relative_to(snapshot)) for p in snapshot.rglob("*") if p.is_file()} == set(files), "snapshot inventory"
    total = 0
    for name, info in sorted(files.items()):
        p = snapshot / name
        assert p.is_file() and p.stat().st_size == info["size"], f"size: {name}"
        kind, expected = info["hash_kind"], info["hash"]
        assert kind in ("sha256", "git-sha1"), f"hash kind: {name}"
        h = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
        if kind == "git-sha1":
            h.update(f"blob {info['size']}\0".encode())
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        assert h.hexdigest() == expected, f"digest: {name}"
        total += info["size"]
    return {"verified": True, "revision": REVISION, "files": len(files), "bytes": total,
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    print(json.dumps(verify(args.snapshot, json.loads(args.manifest.read_text())), sort_keys=True))
