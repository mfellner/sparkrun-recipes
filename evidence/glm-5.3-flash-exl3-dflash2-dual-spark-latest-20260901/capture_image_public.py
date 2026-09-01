#!/usr/bin/env python3
"""Capture public GHCR and per-host image identity without storing tokens."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

REPOSITORY = "ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks"
TAG = "c190db1"
DIGEST = "sha256:4f30fba4248ed7d78dddda37d9ffc0fea5cc8c567c5dd98042ad808efdde9791"
REFERENCE = f"{REPOSITORY}@{DIGEST}"
SOURCE_REVISION = "c190db1ae17ba8dff20129ed1f308d10c63cf37d"
WORKER = "192.168.178.46"


def run(argv: list[str]) -> dict:
    process = subprocess.run(argv, text=True, capture_output=True, timeout=120)
    return {"argv": argv, "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    token_url = "https://ghcr.io/token?scope=repository:mfellner/glm-5.3-flash-2x-dgx-sparks:pull&service=ghcr.io"
    token_request = urllib.request.Request(token_url, headers={"User-Agent": "sparkrun-recipes-image-audit/1"})
    with urllib.request.urlopen(token_request, timeout=30) as response:
        token_status = response.status
        token = json.load(response)["token"]
    manifest_url = f"https://ghcr.io/v2/mfellner/glm-5.3-flash-2x-dgx-sparks/manifests/{TAG}"
    manifest_request = urllib.request.Request(
        manifest_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.v2+json",
            "User-Agent": "sparkrun-recipes-image-audit/1",
        },
    )
    with urllib.request.urlopen(manifest_request, timeout=30) as response:
        manifest_body = response.read()
        manifest_status = response.status
        registry_digest = response.headers.get("Docker-Content-Digest")
        media_type = response.headers.get("Content-Type")

    local = run(["docker", "image", "inspect", REFERENCE])
    worker = run(["ssh", WORKER, f"docker image inspect {REFERENCE}"])
    package = run(["gh", "api", "/user/packages/container/glm-5.3-flash-2x-dgx-sparks"])
    for label, row in (("local", local), ("worker", worker), ("package", package)):
        if row["returncode"] != 0:
            raise RuntimeError(f"{label} command failed: {row['stderr']}")

    record = {
        "schema": 2,
        "captured_at_epoch": time.time(),
        "capture_script_sha256": script_sha,
        "repository": REPOSITORY,
        "tag": TAG,
        "reference": REFERENCE,
        "expected_digest": DIGEST,
        "source_revision": SOURCE_REVISION,
        "anonymous_registry": {
            "token_url": token_url,
            "token_http": token_status,
            "manifest_url": manifest_url,
            "manifest_http": manifest_status,
            "content_type": media_type,
            "docker_content_digest": registry_digest,
            "manifest_body_base64": base64.b64encode(manifest_body).decode(),
            "manifest_body_sha256": hashlib.sha256(manifest_body).hexdigest(),
            "manifest_size": len(manifest_body),
        },
        "local_inspect": local,
        "worker_inspect": worker,
        "package_api": package,
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"digest": registry_digest, "manifest_http": manifest_status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
