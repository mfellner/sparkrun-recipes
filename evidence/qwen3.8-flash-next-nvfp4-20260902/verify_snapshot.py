#!/usr/bin/env python3
"""Verify an immutable Hugging Face safetensors snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-total-bytes", type=int, required=True)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve(strict=True)
    index = snapshot / "model.safetensors.index.json"
    payload = json.loads(index.read_text())
    shards = sorted(set(payload["weight_map"].values()))
    missing = [name for name in shards if not (snapshot / name).is_file()]
    broken = [str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_symlink() and not path.exists()]
    resolved_bytes = sum((snapshot / name).stat().st_size for name in shards if (snapshot / name).is_file())
    metadata_bytes = int(payload.get("metadata", {}).get("total_size", -1))
    revision_ok = snapshot.name == args.expected_revision
    passed = (
        revision_ok
        and len(shards) == args.expected_shards
        and not missing
        and not broken
        and resolved_bytes == args.expected_total_bytes
        and metadata_bytes == args.expected_total_bytes
    )
    result = {
        "snapshot": str(snapshot),
        "revision": snapshot.name,
        "revision_ok": revision_ok,
        "shard_count": len(shards),
        "missing_shards": missing,
        "broken_symlinks": broken,
        "resolved_shard_bytes": resolved_bytes,
        "index_metadata_total_size": metadata_bytes,
        "index_sha256": sha256(index),
        "config_sha256": sha256(snapshot / "config.json"),
        "passed": passed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
