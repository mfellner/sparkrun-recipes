#!/usr/bin/env python3
"""Stage a locally built dense-EXL3 H3 pair in the HF cache (GLM53_MODEL_PRESET=dense-h3).

    stage_dense_h3.py draft  --built <packaged draft dir> --hub <hub> --repo local/<name>
    stage_dense_h3.py target --overlay <hub>/<target repo>/snapshots/<build dir> --hub <hub> \
        --draft-repo local/<name> --draft-rev <rev> --ref <refs name> --sources <json>

Each command prints the staged revision: a SHA-1 over the snapshot's file
identities (an HF blob name for a cache link, the link target for a linked
directory, the SHA-256 of the bytes otherwise), so it is the immutable 40-hex id tools/pack_profile.py requires
and identical bytes always land on the same id. The target gains the
glm53_profile block that names its paired draft; its revision is written to
refs/<ref>, never refs/main, so the ordinary pack path keeps its own snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

PROFILE = "dense-exl3-h3-dflash2-6bpw"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def revision(snapshot: Path) -> str:
    lines = []
    for entry in sorted(snapshot.iterdir()):
        if entry.name.startswith("."):
            continue
        resolved = entry.resolve(strict=True)
        if entry.is_symlink() and resolved.is_dir():
            # a linked sidecar directory of the base snapshot: its relative target names it
            lines.append(f"{entry.name} link:{os.readlink(entry)}\n")
            continue
        if not resolved.is_file():
            raise ValueError(f"not a regular file: {entry}")
        if entry.is_symlink() and resolved.parent.name == "blobs":
            ident = "blob:" + resolved.name
        else:
            ident = "sha256:" + file_sha256(resolved)
        lines.append(f"{entry.name} {ident}\n")
    return hashlib.sha1("".join(lines).encode()).hexdigest()


def publish(staging: Path, ref_file: Path) -> str:
    """Rename a staging dir to snapshots/<rev> and point ref_file at it."""
    rev = revision(staging)
    final = staging.parent / rev
    if final.exists():
        shutil.rmtree(staging)  # same identity: the staged bytes are already there
    else:
        staging.rename(final)
    ref_file.parent.mkdir(parents=True, exist_ok=True)
    ref_file.write_text(rev)
    return rev


def repo_dir(hub: Path, repo: str) -> Path:
    return hub / ("models--" + repo.replace("/", "--"))


def stage_draft(built: Path, hub: Path, repo: str) -> str:
    files = [p for p in sorted(built.iterdir()) if p.is_file() and not p.name.startswith(".")]
    if not {"config.json", "model.safetensors"} <= {p.name for p in files}:
        raise ValueError(f"packaged draft is missing config.json/model.safetensors: {built}")
    root = repo_dir(hub, repo)
    staging = root / "snapshots" / ".glm53-staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for p in files:
        shutil.copyfile(p, staging / p.name)
    return publish(staging, root / "refs" / "main")


def stage_target(overlay: Path, hub: Path, draft_repo: str, draft_rev: str, ref: str,
                 sources: dict) -> str:
    if overlay.parent.name != "snapshots":
        raise ValueError(f"overlay must be built inside the target repo's snapshots/: {overlay}")
    draft_config = repo_dir(hub, draft_repo) / "snapshots" / draft_rev / "config.json"
    config_path = overlay / "config.json"
    cfg = json.loads(config_path.read_text())
    if not cfg.get("quantization_config", {}).get("non_routed_exl3", {}).get("layers"):
        raise ValueError(f"not a dense-EXL3 overlay pack: {overlay}")
    cfg["glm53_profile"] = {
        "name": PROFILE,
        "draft": {"model": draft_repo, "revision": draft_rev,
                  "config_sha256": file_sha256(draft_config)},
        "sources": sources,
    }
    # dense_overlay.py writes config.json as a regular file; replace, never follow a link.
    config_path.unlink()
    config_path.write_text(json.dumps(cfg, indent=2))
    return publish(overlay, overlay.parent.parent / "refs" / ref)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--built", required=True, type=Path)
    d.add_argument("--hub", required=True, type=Path)
    d.add_argument("--repo", required=True)
    t = sub.add_parser("target")
    t.add_argument("--overlay", required=True, type=Path)
    t.add_argument("--hub", required=True, type=Path)
    t.add_argument("--draft-repo", required=True)
    t.add_argument("--draft-rev", required=True)
    t.add_argument("--ref", required=True)
    t.add_argument("--sources", default="{}")
    args = ap.parse_args()
    try:
        if args.cmd == "draft":
            rev = stage_draft(args.built, args.hub, args.repo)
        else:
            rev = stage_target(args.overlay, args.hub, args.draft_repo, args.draft_rev, args.ref,
                               json.loads(args.sources))
    except (OSError, ValueError) as exc:
        print(f"stage_dense_h3: {exc}", file=sys.stderr)
        return 2
    print(rev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
