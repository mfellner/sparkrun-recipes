#!/usr/bin/env python3
"""Independently recompute staged public-artifact claims from raw commands."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = ROOT / "scripts/verify_tensorfold_v171_snapshot.py"
MANIFEST = HERE / "hf-manifest.json"
IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:a8067cd7e14c14fa83d1dbed60261428f6d1737cec4554445573354af040dd7c"
IMAGE_ID = "sha256:549afbcdb787fec95446c207e572acb2791488670061054cc01f526db2fe6f28"
REV = "078455ffe6472f9a52fbc1139f58b9db2881b25c"
HOSTS = ("192.168.178.47", "192.168.178.46")
SNAPSHOT = ("/home/max/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/"
            "snapshots/" + REV)
REMOTE_MANIFEST = "/home/max/.hermes/cache/scratch/tf-v171-hf-manifest.json"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def verify(data: dict) -> dict:
    require(data["schema"] == 1 and set(data["hosts"]) == set(HOSTS), "staging schema/hosts")
    require(data["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest(), "script drift")
    require(data["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), "manifest drift")
    expected_manifest = hashlib.sha256(json.dumps(json.loads(MANIFEST.read_text()), sort_keys=True).encode()).hexdigest()
    for host in HOSTS:
        rows = data["hosts"][host]
        require(set(rows) == {"image", "snapshot"}, "probe inventory")
        for row in rows.values():
            require(row["returncode"] == 0 and not row["stderr"].strip(), "probe error")
            require(0 < row["started_at"] < row["finished_at"] <= data["captured_at"], "probe interval")
        image = rows["image"]
        require(image["argv"] == ["ssh", "-o", "BatchMode=yes", host, "docker", "image", "inspect", IMAGE], "image command")
        raw_image = json.loads(image["stdout"])
        require(len(raw_image) == 1, "image count")
        obj = raw_image[0]
        require(obj["Id"] == IMAGE_ID and IMAGE in obj["RepoDigests"] and obj["Architecture"] == "arm64", "image identity")
        require(obj["Config"]["Labels"]["tf.patches"] == "1692d2df78d2", "patchset")
        snap = rows["snapshot"]
        require(snap["argv"] == ["ssh", "-o", "BatchMode=yes", host, "python3", "-",
                                  "--snapshot", SNAPSHOT, "--manifest", REMOTE_MANIFEST], "snapshot command")
        d = json.loads(snap["stdout"])
        require(d == {"verified": True, "revision": REV, "files": 97, "bytes": 175716135696,
                      "manifest_sha256": expected_manifest}, "snapshot identity/contents")
    return {"passed": True, "hosts": len(HOSTS), "snapshot_files_per_host": 97,
            "image_id": IMAGE_ID, "revision": REV}


if __name__ == "__main__":
    print(json.dumps(verify(json.loads((HERE / "staging.json").read_text())), indent=2))
