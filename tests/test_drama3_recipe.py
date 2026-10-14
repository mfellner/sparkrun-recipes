"""Immutable source/build/recipe boundaries for the native single-Spark profile."""
import hashlib
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mods/drama3-gb10"


def test_runtime_manifest_matches_all_packaged_bytes():
    entries = dict(line.split("  ", 1)[::-1] for line in (MOD / "manifest.sha256").read_text().splitlines())
    required = {"Dockerfile", "requirements-lock.txt", "requirements-runtime.txt", "contract.py", "server.py", "probe.py", "run.sh", "model-manifest.json", "NOTICE", "FISH_LICENSE.txt", "ADAPTER_LICENSE.txt", "default-reference.wav", "default-reference.txt", "reference-provenance.json"}
    assert set(entries) == required
    for name, expected in entries.items():
        assert hashlib.sha256((MOD / name).read_bytes()).hexdigest() == expected


def test_recipe_is_single_native_pinned_lan_profile():
    recipe = yaml.safe_load((ROOT / "recipes/fish-audio-drama3-dgx-spark.yaml").read_text())
    assert recipe["model_revision"] == "dd64a85cd82e9521f7202610ede97b367fc83946"
    assert recipe["max_nodes"] == recipe["defaults"]["tensor_parallel"] == 1
    assert re.fullmatch(r"ghcr.io/mfellner/[^@]+@sha256:[0-9a-f]{64}", recipe["container"])
    assert recipe["defaults"]["host"] == "192.168.178.70"
    assert recipe["defaults"]["port"] == 8010
    assert "vllm serve" not in recipe["command"]
    assert recipe["env"]["FISH_DRAMA3_FP8_EXPERTS"] == "1"
    assert recipe["env"]["FISH_DRAMA3_CUDA_GRAPH"] == "1"
    executor = recipe["executor_config"]
    assert executor["privileged"] is False
    assert executor["user"] == "1000:1000"
    assert executor["ipc"] == "private"
    assert executor["security_opt"] == ["no-new-privileges"]
    entry = recipe["distribution_config"]["models"]["entries"][0]
    assert entry["revision"] == recipe["model_revision"]


def test_model_and_default_reference_manifests():
    manifest = json.loads((MOD / "model-manifest.json").read_text())
    assert manifest["revision"] == "dd64a85cd82e9521f7202610ede97b367fc83946"
    shards = [row for row in manifest["files"] if row["path"].endswith(".safetensors")]
    assert len(shards) == 8
    assert sum(row["size"] for row in shards) == 39603574344
    codec = next(row for row in manifest["files"] if row["path"] == "codec.pth")
    assert codec["size"] == 1871099728
    reference = json.loads((MOD / "reference-provenance.json").read_text())
    assert reference["sha256"] == hashlib.sha256((MOD / "default-reference.wav").read_bytes()).hexdigest()
    assert reference["transcript"] == (MOD / "default-reference.txt").read_text().strip()


def test_public_build_does_not_include_weights_or_mutable_fish_source():
    dockerfile = (MOD / "Dockerfile").read_text()
    assert "@sha256:450d11555d20ac8ebbbc13ebf17589c2bd42869171a90179ce7098b4a5e64c6a" in dockerfile
    assert "git fetch --depth 1 origin 1ff3c3ed2f80d3106a83909dd25b83129c3d816f" in dockerfile
    assert "COPY . " not in dockerfile
    assert "snapshot_download" not in dockerfile and "codec.pth" not in dockerfile
    assert "--no-deps -r /opt/drama/requirements-lock.txt" in dockerfile
