#!/usr/bin/env python3
"""Stage only the pinned Drama3 manifest, resumably. SPDX-License-Identifier: MIT."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "mods/drama3-gb10/model-manifest.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(8 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def download(entry, directory, revision):
    file = directory / entry["path"]
    if file.is_file() and file.stat().st_size == entry["size"] and digest(file) == entry["sha256"]:
        print("verified cached", file.name, flush=True)
        return
    part = file.with_name(file.name + ".part")
    url = f"https://huggingface.co/fishaudio/drama-3/resolve/{revision}/{file.name}"
    subprocess.run(["curl", "--fail", "--location", "--retry", "10", "--retry-all-errors", "--connect-timeout", "30", "--max-time", "7200", "--continue-at", "-", "--output", str(part), url], check=True)
    if part.stat().st_size != entry["size"] or digest(part) != entry["sha256"]:
        raise ValueError(f"size/checksum mismatch: {file.name}; retained .part for diagnosis")
    part.replace(file)
    print("verified downloaded", file.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/huggingface")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["revision"] != "dd64a85cd82e9521f7202610ede97b367fc83946":
        raise ValueError("unexpected model revision")
    snapshot = args.cache / "hub/models--fishaudio--drama-3/snapshots" / manifest["revision"]
    snapshot.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda entry: download(entry, snapshot, manifest["revision"]), manifest["files"]))
    print(json.dumps({"passed": True, "snapshot": str(snapshot), "revision": manifest["revision"], "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
