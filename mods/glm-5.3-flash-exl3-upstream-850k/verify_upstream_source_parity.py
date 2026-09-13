#!/usr/bin/env python3
"""Fail-closed byte parity against the exact pinned upstream archive."""
from __future__ import annotations

import io
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

REVISION = "f906ee990596486e10ddbe381efa6f0e496f77e3"
REPOSITORY = "MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks"
ARCHIVE_URL = f"https://github.com/{REPOSITORY}/archive/{REVISION}.tar.gz"
UPSTREAM = Path(__file__).resolve().parent / "upstream"


def local_files() -> dict[str, bytes]:
    return {
        path.relative_to(UPSTREAM).as_posix(): path.read_bytes()
        for path in UPSTREAM.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def archive_files(payload: bytes) -> dict[str, bytes]:
    rows: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 2:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            relative = PurePosixPath(*parts[1:]).as_posix()
            if relative in rows:
                raise RuntimeError(f"duplicate archive member: {relative}")
            rows[relative] = stream.read()
    return rows


def main() -> int:
    with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"source archive HTTP status {response.status}")
        expected = archive_files(response.read())
    actual = local_files()
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        print(
            "source parity FAIL: relative file set mismatch; "
            f"missing={missing[:10]!r} extra={extra[:10]!r}",
            file=sys.stderr,
        )
        return 1
    mismatches = [name for name in sorted(actual) if actual[name] != expected[name]]
    if mismatches:
        print(f"source parity FAIL: byte mismatch: {mismatches[0]}", file=sys.stderr)
        return 1
    print(f"source parity PASS: {REPOSITORY}@{REVISION} files={len(actual)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
