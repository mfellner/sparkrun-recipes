"""Apply only to the byte-verified pinned Fish tree; no model imports.

SPDX-License-Identifier: MIT. Source mismatch fails before any write. Not idempotent:
applying twice is a mismatch, not permission to trust a previously changed tree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
BASE = Path("fish_speech/models/text2semantic/fish_omni")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(root, verify_only=False):
    root = root.resolve(strict=True)
    pins = json.loads((HERE / "source-pin.json").read_text())
    patch = HERE / "native.patch"
    if digest(patch) != pins["patch_sha256"]:
        raise ValueError("patch checksum mismatch")
    expected_source = {**pins["before"], **pins["after"]} if verify_only else pins["before"]
    for name, expected in expected_source.items():
        path = root / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError(f"pinned Fish source mismatch: {name}")
    target = root / BASE / "native_perf.py"
    if verify_only:
        if target.is_symlink() or digest(target) != digest(HERE / "native_perf.py"):
            raise ValueError("installed native_perf runtime mismatch")
        print(json.dumps({"verified": True, "source_revision": pins["revision"]}))
        return
    if target.exists():
        raise ValueError("native_perf.py already exists")
    for args in (["--check"], []):
        subprocess.run(["git", "apply", *args, str(patch)], cwd=root, check=True)
    for name, expected in pins["after"].items():
        if digest(root / name) != expected:
            raise ValueError(f"patched source readback mismatch: {name}")
    shutil.copyfile(HERE / "native_perf.py", target)
    print(json.dumps({"applied": True, "source_revision": pins["revision"],
                      "source_after": pins["after"], "runtime_sha256": digest(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("fish_root", type=Path)
    args = parser.parse_args()
    apply(args.fish_root, args.verify_only)
