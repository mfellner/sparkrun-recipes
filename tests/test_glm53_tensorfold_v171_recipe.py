"""Contracts for the upstream v1.7.1 TensorFold TP2 profile, without rewriting the historical run."""
from pathlib import Path
import hashlib
import json
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/glm-5.3-flash-exl3-tensorfold-dual-spark-1m-v171.yaml"
OLD = ROOT / "recipes/glm-5.3-flash-exl3-tensorfold-dual-spark-1m.yaml"
MOD = ROOT / "mods/glm53-tensorfold-1m"
MODEL = "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold"
MODEL_SHA = "078455ffe6472f9a52fbc1139f58b9db2881b25c"
DRAFT_SHA = "bf582e4eacc1810f76656d1811693ff6c6737d2a"
IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:a8067cd7e14c14fa83d1dbed60261428f6d1737cec4554445573354af040dd7c"
SOURCE = "68ebd67b5326974b8004009e202268b1fa7c551d"


def test_v171_immutable_selected_identity_and_unchanged_historical_recipe():
    r = yaml.safe_load(RECIPE.read_text())
    old = yaml.safe_load(OLD.read_text())
    assert r["name"] == "glm-5.3-flash-exl3-tensorfold-dual-spark-1m-v171"
    assert r["runtime"] == "vllm-distributed"  # placement shim; TensorFold serves
    assert r["runtime_version"] == "tensorfold-0.6.0"
    assert r["min_nodes"] == r["max_nodes"] == 2
    assert r["model"] == MODEL and r["model_revision"] == MODEL_SHA
    assert r["container"] == IMAGE
    assert r["metadata"]["source_revision"] == SOURCE
    assert r["distribution_config"]["models"]["entries"] == [
        {"name": MODEL, "revision": MODEL_SHA, "target": [-1]},
        {"name": "incoai/GLM-5.3-Flash-DFlash2", "revision": DRAFT_SHA, "target": [-1]},
    ]
    assert r["command"].endswith(
        f"/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/{MODEL_SHA}"
    )
    assert r["mods"] == old["mods"] == ["../mods/glm53-tensorfold-1m"]
    # Never rewrite the accepted TR3 profile, which the 2026-10-01 receipts bind.
    assert old["model"] == "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw"
    assert old["model_revision"] == "9eaebb7c4e96d983dcd538e18624622ba5b820a8"
    assert old["container"].endswith("@sha256:22789f0cb3dc308f0b2ce52a33961b88bd624af1725e91e8aba0a74a671bb969")


def test_v171_explicit_tp2_c4_upstream_defaults_and_fabric():
    r = yaml.safe_load(RECIPE.read_text())
    e = r["env"]
    expected = {
        "TF_CONTEXT": "1048576", "TF_PARALLEL": "4", "TF_MAX_TOKENS": "32768",
        "TF_GLM_KV": "fp8", "TF_GLM_DENSE": "q4", "TF_GLM_MTP": "auto",
        "TF_GLM_COMM": "roce", "TF_GLM_DFLASH_POLICY": "fnc7:0.3",
        "TF_GLM_CACHE_GIB": "12.5", "TENSORFOLD_MEMORY_RESERVE_GIB": "14.5",
        "TF_GLM_MULTI_LONE": "0", "TF_GLM_MULTI_WINDOW": "32",
        "TF_GLM_CACHE_ENTRIES": "32", "TF_GLM_DISPLAY_KV_MIB": "0",
        "TF_GLM_ASSISTANT_ENDS": "1", "TF_GLM_CLEAR_THINKING": "0",
        "TF_GLM_STREAM_SMOOTH": "1", "TF_GLM_STREAM_SMOOTH_MS": "400",
        "TF_GLM_FILL_BUDGET_MS": "200", "TF_GLM_FILL_DRAFTS": "1",
        "TF_GLM_MULTI_PREFILL": "1", "TF_GLM_HC_SPLIT": "1",
        "TF_ROCE_MAX_KB": "512", "TF_ROCE_WAIT_S": "300",
        "TF_THINKING": "1", "TF_VISION": "1", "TF_VISION_URLS": "0",
        "TF_SERVED_NAME": "GLM-5.3-Flash-EXL3", "TF_PORT": "8000",
    }
    for key, value in expected.items():
        assert str(e[key]) == value, key
    assert e["TF_DRAFTER_SNAPSHOT"].endswith("/" + DRAFT_SHA)
    assert (e["TF_FABRIC_MASTER"], e["TF_FABRIC_WORKER"]) == ("192.168.3.72", "192.168.3.183")
    assert (e["TF_FABRIC_MASTER_SECONDARY"], e["TF_FABRIC_WORKER_SECONDARY"]) == ("192.168.2.72", "192.168.2.183")
    assert r["defaults"]["tensor_parallel"] == 2
    assert r["defaults"]["port"] == 8000
    assert r["metadata"]["context_length"] == 1048576
    assert r["executor_config"]["entrypoint"] == ""
    assert "nofile=65536:65536" in r["executor_config"]["ulimit"]
    assert r["executor_config"]["cap_add"] == ["IPC_LOCK"]


def test_v171_reuses_byte_identical_two_rail_adapter():
    lines = (MOD / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 2
    for row in lines:
        digest, name = row.split("  ", 1)
        assert hashlib.sha256((MOD / name).read_bytes()).hexdigest() == digest
    for script in ("serve.sh", "run.sh"):
        subprocess.run(["bash", "-n", str(MOD / script)], check=True)


def test_v171_manifest_pins_all_shards_and_rejects_incomplete_snapshot(tmp_path):
    manifest = json.loads((ROOT / "evidence/glm53-tensorfold-20261005/hf-manifest.json").read_text())
    assert manifest["id"] == MODEL and manifest["sha"] == MODEL_SHA
    files = manifest["files"]
    assert len(files) == 97
    assert len([name for name in files if name.startswith("model-") and name.endswith(".safetensors")]) == 83
    assert all(row["hash_kind"] in ("sha256", "git-sha1") and row["hash"] for row in files.values())
    result = subprocess.run(["python3", str(ROOT / "scripts/verify_tensorfold_v171_snapshot.py"),
                             "--snapshot", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and "snapshot inventory" in result.stderr


def test_v171_license_disclosure_does_not_inherit_historical_tr3_terms():
    readme = (ROOT / "README.md").read_text()
    license_section = readme.split("## License\n", 1)[1]
    assert "historical GLM-5.3-Flash-EXL3-TR3-4bpw checkpoint is reported\nunder ShapleyMCG License 1.0" in license_section
    assert "newer public TensorFold checkpoint has\nconflicting publisher statements" in license_section
    assert "MIT in its pinned card and bundled LICENSE" in license_section
    assert "Apache-2.0 in the launcher NOTICE" in license_section
    assert "DFlash2 reports CC\nBY-NC-ND 4.0" in license_section
