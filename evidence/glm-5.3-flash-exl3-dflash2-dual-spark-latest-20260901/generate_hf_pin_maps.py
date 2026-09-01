#!/usr/bin/env python3
"""Generate deterministic Hugging Face file maps for pin provenance."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "raw/hf-pins.json"
SOURCES = {
    "model_old": ("Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw", "25a44fdbf16862a46b7cc9921142c6c81350af2f"),
    "model_pinned": ("Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw", "024db9f7e9871e8efdf21538ba55af7442be3cd5"),
    "draft_old": ("incoai/GLM-5.3-Flash-DFlash2", "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"),
    "draft_pinned": ("incoai/GLM-5.3-Flash-DFlash2", "bf582e4eacc1810f76656d1811693ff6c6737d2a"),
}
ORIGIN = ("brandonmusic/GLM-5.3-Flash-tr3-4bpw", "5ab363a8dcf6405955fd5f99671e01a1c9fb124b")
NON_SERVING_FILES = {".gitattributes", "README.md", "MIRROR.json", "ORIGINAL_MODEL_CARD.md"}
NON_SERVING_PREFIXES = ("assets/", "docs/", "provenance/", "receipts/")
CORE_FILES = {
    "chat_template.jinja",
    "config.json",
    "exl3-mcg-storage-abi.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "quantization/recipe.json",
    "quantization_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
SHARD_RX = re.compile(r"model-\d{5}-of-\d{5}[.]safetensors$")


def repo_path(repo: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in repo.split("/"))


def file_path(path: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "sparkrun-recipes-pin-audit/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_json(url: str) -> dict[str, Any]:
    return json.loads(fetch_bytes(url))


def api_map(repo: str, revision: str) -> dict[str, Any]:
    obj = fetch_json(f"https://huggingface.co/api/models/{repo_path(repo)}/revision/{revision}?blobs=true")
    rows = []
    for sibling in sorted(obj.get("siblings", []), key=lambda row: row["rfilename"]):
        lfs = sibling.get("lfs") or {}
        rows.append(
            {
                "path": sibling["rfilename"],
                "size": sibling.get("size"),
                "blob_id": sibling.get("blobId"),
                "lfs_sha256": lfs.get("sha256"),
                "lfs_size": lfs.get("size"),
            }
        )
    if obj["sha"] != revision:
        raise RuntimeError(f"{repo}: requested {revision}, resolved {obj['sha']}")
    return {"repo": repo, "revision": revision, "files": rows}


def identity(row: dict[str, Any]) -> tuple[Any, ...]:
    if row.get("sha256"):
        return (row.get("size"), "sha256", row["sha256"])
    if row.get("lfs_sha256"):
        return (row.get("lfs_size") or row.get("size"), "lfs_sha256", row["lfs_sha256"])
    return (row.get("size"), "blob_id", row.get("blob_id"))


def file_map(rows: list[dict[str, Any]], serving_only: bool = False) -> dict[str, tuple[Any, ...]]:
    return {
        row["path"]: identity(row)
        for row in rows
        if not serving_only
        or (
            row["path"] not in NON_SERVING_FILES
            and not row["path"].startswith(NON_SERVING_PREFIXES)
        )
    }


def diff_maps(left: dict[str, tuple[Any, ...]], right: dict[str, tuple[Any, ...]]) -> dict[str, Any]:
    left_paths = set(left)
    right_paths = set(right)
    return {
        "left_count": len(left),
        "right_count": len(right),
        "added": sorted(right_paths - left_paths),
        "removed": sorted(left_paths - right_paths),
        "changed": sorted(path for path in left_paths & right_paths if left[path] != right[path]),
        "equal": left == right,
    }


def mirror_origin_payload(mirror: dict[str, Any]) -> dict[str, Any]:
    origin_repo, origin_revision = ORIGIN
    manifest = fetch_json(f"https://huggingface.co/{repo_path(origin_repo)}/resolve/{origin_revision}/MANIFEST.json")
    manifest_rows = {row["path"]: row for row in manifest["files"]}
    receipt_paths = sorted(path for path in manifest_rows if path.startswith(".materialization/shards/"))

    def receipt(path: str) -> dict[str, Any]:
        row = fetch_json(f"https://huggingface.co/{repo_path(origin_repo)}/resolve/{origin_revision}/{file_path(path)}")
        if row.get("complete") is not True:
            raise RuntimeError(f"incomplete materialization receipt: {path}")
        return {"path": row["shard"], "size": row["shard_bytes"], "sha256": row["shard_sha256"], "receipt_path": path}

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        origin_shards = sorted(pool.map(receipt, receipt_paths), key=lambda row: row["path"])
    origin_payload = {row["path"]: row for row in origin_shards}
    for path in sorted(CORE_FILES):
        row = manifest_rows[path]
        origin_payload[path] = {"path": path, "size": row["bytes"], "sha256": row["sha256"], "source": "origin MANIFEST.json"}

    mirror_rows = {row["path"]: row for row in mirror["files"]}
    mirror_payload: dict[str, dict[str, Any]] = {}
    for path, row in mirror_rows.items():
        if SHARD_RX.fullmatch(path):
            mirror_payload[path] = {"path": path, "size": row["lfs_size"] or row["size"], "sha256": row["lfs_sha256"], "source": "mirror LFS map"}
    mirror_repo, mirror_revision = SOURCES["model_pinned"]
    for path in sorted(CORE_FILES):
        raw = fetch_bytes(f"https://huggingface.co/{repo_path(mirror_repo)}/resolve/{mirror_revision}/{file_path(path)}")
        mirror_payload[path] = {"path": path, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "source": "mirror immutable raw file"}

    left = {path: identity(row) for path, row in mirror_payload.items()}
    right = {path: identity(row) for path, row in origin_payload.items()}
    comparison = diff_maps(left, right)
    if not comparison["equal"]:
        raise RuntimeError(f"mirror/origin logical serving payload differs: {comparison}")
    return {
        "repo": origin_repo,
        "revision": origin_revision,
        "manifest": {
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "core_files": sorted(CORE_FILES),
        "mirror_logical_payload": [mirror_payload[path] for path in sorted(mirror_payload)],
        "origin_logical_payload": [origin_payload[path] for path in sorted(origin_payload)],
        "comparison": comparison,
    }


def main() -> int:
    maps = {name: api_map(*source) for name, source in SOURCES.items()}
    full = {name: file_map(data["files"]) for name, data in maps.items()}
    serving = {name: file_map(data["files"], serving_only=True) for name, data in maps.items()}
    comparisons = {
        "model_old_to_pinned_full": diff_maps(full["model_old"], full["model_pinned"]),
        "model_old_to_pinned_serving": diff_maps(serving["model_old"], serving["model_pinned"]),
        "draft_old_to_pinned_full": diff_maps(full["draft_old"], full["draft_pinned"]),
        "draft_old_to_pinned_serving": diff_maps(serving["draft_old"], serving["draft_pinned"]),
    }
    if not comparisons["model_old_to_pinned_serving"]["equal"]:
        raise RuntimeError("main mirror serving payload changed from prior pin")
    if comparisons["draft_old_to_pinned_serving"]["equal"]:
        raise RuntimeError("draft serving payload unexpectedly unchanged")
    origin = mirror_origin_payload(maps["model_pinned"])
    record = {
        "schema": 3,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_revision": "c190db1ae17ba8dff20129ed1f308d10c63cf37d",
        "identity_rule": "size plus LFS SHA-256 or git blob ID; origin chain uses size plus SHA-256",
        "non_serving_files": sorted(NON_SERVING_FILES),
        "non_serving_prefixes": list(NON_SERVING_PREFIXES),
        "sources": maps,
        "comparisons": comparisons,
        "origin_equivalence": origin,
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({**comparisons, "model_pinned_to_origin_serving": origin["comparison"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
