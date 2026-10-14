#!/usr/bin/env python3
"""Anonymous GHCR digest/config round-trip; no saved credentials. SPDX-License-Identifier: MIT."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def probe(image):
    repository, digest = image.removeprefix("ghcr.io/").split("@", 1)
    request = urllib.request.Request("https://ghcr.io/token?" + urllib.parse.urlencode({"service": "ghcr.io", "scope": "repository:" + repository + ":pull"}))
    with urllib.request.urlopen(request, timeout=60) as response:
        # This is an anonymous short-lived pull capability, never persisted/logged.
        token = json.load(response)["token"]
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"}
    url = "https://ghcr.io/v2/" + repository + "/manifests/" + digest
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        raw = response.read()
        status = response.status
        resolved = response.headers.get("Docker-Content-Digest")
    if "sha256:" + hashlib.sha256(raw).hexdigest() != digest or resolved != digest:
        raise ValueError("registry manifest differs from requested digest")
    manifest = json.loads(raw)
    config_url = "https://ghcr.io/v2/" + repository + "/blobs/" + manifest["config"]["digest"]
    with urllib.request.urlopen(urllib.request.Request(config_url, headers=headers), timeout=60) as response:
        config_bytes = response.read()
    if "sha256:" + hashlib.sha256(config_bytes).hexdigest() != manifest["config"]["digest"]:
        raise ValueError("registry config digest mismatch")
    config = json.loads(config_bytes)
    if config["architecture"] != "arm64" or config["os"] != "linux":
        raise ValueError("not a Linux ARM64 image")
    return dict(image=image, anonymous=True, status=status, manifest_digest=resolved, config_digest=manifest["config"]["digest"], architecture=config["architecture"], os=config["os"], labels=config["config"].get("Labels", {}), user=config["config"].get("User"), entrypoint=config["config"].get("Entrypoint"), compressed_layer_bytes=sum(row["size"] for row in manifest["layers"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = probe(args.image)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
