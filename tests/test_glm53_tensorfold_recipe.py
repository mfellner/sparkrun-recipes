"""Contracts for the pinned TensorFold dual-Spark adapter (CPU-only)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/glm-5.3-flash-exl3-tensorfold-dual-spark-1m.yaml"
MOD = ROOT / "mods/glm53-tensorfold-1m"
MODEL_SHA = "9eaebb7c4e96d983dcd538e18624622ba5b820a8"
DRAFT_SHA = "bf582e4eacc1810f76656d1811693ff6c6737d2a"
IMAGE_DIGEST = "sha256:22789f0cb3dc308f0b2ce52a33961b88bd624af1725e91e8aba0a74a671bb969"


def test_pinned_identity_and_cluster_contract():
    recipe = yaml.safe_load(RECIPE.read_text())
    assert recipe["min_nodes"] == recipe["max_nodes"] == 2
    assert recipe["runtime"] == "vllm-distributed"  # orchestration shim, NOT model runtime
    assert recipe["runtime_version"] == "tensorfold-0.6.0"
    assert recipe["container"].endswith("@" + IMAGE_DIGEST)
    assert recipe["metadata"]["source_revision"] == "978b2252059069b3b4b84f0f7eeb73bc17f28d3f"
    assert recipe["model_revision"] == MODEL_SHA
    assert f"snapshots/{MODEL_SHA}" in recipe["command"]
    entries = {entry["name"]: entry["revision"] for entry in recipe["distribution_config"]["models"]["entries"]}
    assert entries == {
        "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw": MODEL_SHA,
        "incoai/GLM-5.3-Flash-DFlash2": DRAFT_SHA,
    }
    assert recipe["mods"] == ["../mods/glm53-tensorfold-1m"]
    assert recipe["executor_config"]["entrypoint"] == ""


def test_upstream_default_profile_and_explicit_fabric():
    recipe = yaml.safe_load(RECIPE.read_text())
    env = recipe["env"]
    expected = {
        "TF_CONTEXT": "1048576", "TF_PARALLEL": "4", "TF_MAX_TOKENS": "32768",
        "TF_GLM_KV": "fp8", "TF_GLM_DENSE": "q4", "TF_GLM_MTP": "auto",
        "TF_GLM_COMM": "roce", "TF_GLM_DFLASH_POLICY": "fnc7:0.3",
        "TF_GLM_CACHE_GIB": "12.5", "TENSORFOLD_MEMORY_RESERVE_GIB": "14.5",
        "TF_GLM_MULTI_PREFILL": "1", "TF_GLM_HC_SPLIT": "1",
        "TF_ROCE_MAX_KB": "512", "TF_ROCE_WAIT_S": "20",
        "TF_THINKING": "1", "TF_VISION": "1", "TF_VISION_URLS": "0",
        "TF_SERVED_NAME": "GLM-5.3-Flash-EXL3", "TF_PORT": "8000",
    }
    for key, value in expected.items():
        assert str(env[key]) == value, key
    assert env["TF_DRAFTER_SNAPSHOT"].endswith("/" + DRAFT_SHA)
    assert env["TF_FABRIC_MASTER"] == "192.168.3.72"
    assert env["TF_FABRIC_WORKER"] == "192.168.3.183"
    assert env["TF_FABRIC_MASTER_SECONDARY"] == "192.168.2.72"
    assert env["TF_FABRIC_WORKER_SECONDARY"] == "192.168.2.183"
    assert recipe["defaults"]["served_model_name"] == env["TF_SERVED_NAME"]
    assert recipe["defaults"]["port"] == int(env["TF_PORT"])
    assert recipe["metadata"]["context_length"] == int(env["TF_CONTEXT"])


def test_mod_manifest_is_complete_and_shell_syntax_is_valid():
    manifest = (MOD / "SHA256SUMS").read_text().splitlines()
    names = set()
    for line in manifest:
        digest, name = line.split("  ", 1)
        assert name not in names
        names.add(name)
        assert hashlib.sha256((MOD / name).read_bytes()).hexdigest() == digest
    assert names == {"serve.sh", "fabric.py"}
    for name in ("run.sh", "serve.sh"):
        subprocess.run(["bash", "-n", str(MOD / name)], check=True)


@pytest.mark.parametrize("args", [
    [], ["--nnodes", "1", "--node-rank", "0", "--master-addr", "192.168.178.47", "--master-port", "25000"],
    ["--nnodes", "2", "--node-rank", "1", "--master-addr", "192.168.178.47", "--master-port", "25000"],
    ["--nnodes", "2", "--node-rank", "0", "--master-addr", "192.168.178.47", "--master-port", "25000", "--headless"],
])
def test_wrapper_rejects_missing_or_inconsistent_rank(tmp_path, args):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    result = subprocess.run(["bash", str(MOD / "serve.sh"), str(model), *args], capture_output=True, text=True)
    assert result.returncode != 0


def test_fabric_helper_rejects_invalid_rank_before_hardware():
    spec = importlib.util.spec_from_file_location("tf_fabric", MOD / "fabric.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="invalid rank"):
        module.check("2", "192.168.3.72", "192.168.3.183", "rocep1s0f1", "3")
    with pytest.raises(ValueError):
        module.check("0", "not-an-ip", "192.168.3.72", "rocep1s0f1", "3")


def test_fabric_requires_both_ordered_rails(monkeypatch):
    spec = importlib.util.spec_from_file_location("tf_fabric_two", MOD / "fabric.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("TF_FABRIC_WORKER", "192.168.3.183")
    monkeypatch.setenv("TF_FABRIC_MASTER_SECONDARY", "192.168.2.72")
    monkeypatch.setenv("TF_FABRIC_WORKER_SECONDARY", "192.168.2.183")
    checked = []
    def fake_rail(hca, gid, local, peer):
        checked.append((hca, gid, str(local), str(peer)))
        if hca == "wrongrail":
            raise ValueError("unrelated rail")
        return "enp1s0f1np1"
    monkeypatch.setattr(module, "check_rail", fake_rail)
    assert module.check("0", "192.168.3.72", "192.168.3.72", "rocep1s0f1,roceP2p1s0f1", "3") == "enp1s0f1np1"
    assert checked == [("rocep1s0f1", "3", "192.168.3.72", "192.168.3.183"),
                       ("roceP2p1s0f1", "3", "192.168.2.72", "192.168.2.183")]
    with pytest.raises(ValueError, match="unrelated rail"):
        module.check("0", "192.168.3.72", "192.168.3.72", "rocep1s0f1,wrongrail", "3")
    with pytest.raises(ValueError, match="exactly two"):
        module.check("0", "192.168.3.72", "192.168.3.72", "rocep1s0f1", "3")
    with pytest.raises(ValueError, match="primary fabric"):
        module.check("1", "192.168.3.72", "192.168.3.72", "rocep1s0f1,roceP2p1s0f1", "3")


def test_evidence_verifiers_reject_mutated_http_and_runtime_receipts():
    import copy
    evidence = ROOT / "evidence/glm53-tensorfold-20261001"
    def mutate_privileged(runtime, value):
        row = runtime["hosts"]["192.168.178.47"]["inspect"]
        inspect = json.loads(row["stdout"])
        inspect[0]["HostConfig"]["Privileged"] = value
        row["stdout"] = json.dumps(inspect)

    sys.path.insert(0, str(evidence))
    try:
        spec = importlib.util.spec_from_file_location("tf_verify_runtime", evidence / "verify_runtime.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verify_runtime = module.verify_runtime
        receipt = json.loads((evidence / "receipt.json").read_text())
        runtime = json.loads((evidence / "runtime.json").read_text())
        assert verify_runtime(receipt, runtime)["passed"]
        for mutate in (
            lambda r, t: r["tests"]["proxy"].update(http=502),
            lambda r, t: r["tests"]["direct"]["response"].update(model="other-model"),
            lambda r, t: r["tests"]["proxy"]["response"].update(error={}),
            lambda r, t: t["proxy_status"].update(stdout=t["proxy_status"]["stdout"].replace('"running": true', '"running": "false"', 1)),
            lambda r, t: r["tests"]["concurrent"][0].update(url="http://192.168.178.47:4000/v1/chat/completions"),
            lambda r, t: r["tests"]["vision"].update(url="http://192.168.178.47:4000/v1/chat/completions"),
            lambda r, t: r["tests"]["tool"].update(url="http://192.168.178.47:4000/v1/chat/completions"),
            lambda r, t: r["tests"]["long_needle"].update(url="http://192.168.178.47:4000/v1/chat/completions"),
            lambda r, t: r["health_before"].update(http=503),
            lambda r, t: r["health_after"].update(http=503),
            lambda r, t: r["models"]["direct"]["body"].update(error={"message": "fake"}),
            lambda r, t: r["models"]["proxy"]["body"].update(error={"message": "fake"}),
            lambda r, t: r["health_before"]["body"].update(ok="false"),
            lambda r, t: r["health_after"]["body"].update(ok="false"),
            lambda r, t: t["final_health"]["body"].update(ok="false"),
            lambda r, t: r["health_before"].update(started_at=r["tests"]["direct"]["started_at"] + .001,
                finished_at=r["tests"]["direct"]["started_at"] + .002),
            lambda r, t: r["health_after"].update(started_at=r["tests"]["direct"]["started_at"] + .001,
                finished_at=r["tests"]["direct"]["started_at"] + .002),
            lambda r, t: r["long_tokenize"].update(started_at=r["tests"]["long_needle"]["started_at"] + .001,
                finished_at=r["tests"]["long_needle"]["started_at"] + .002),
            lambda r, t: r["long_tokenize"].update(started_at=r["health_before"]["started_at"] - .002,
                finished_at=r["health_before"]["started_at"] - .001),
            lambda r, t: r["models"]["direct"].update(started_at=r["health_before"]["started_at"] + .001,
                finished_at=r["health_before"]["started_at"] + .002),
            lambda r, t: t["final_health"].update(started_at=t["captured_at"] + .001,
                finished_at=t["captured_at"] + .002),
            lambda r, t: mutate_privileged(t, 0),
            lambda r, t: mutate_privileged(t, None),
            lambda r, t: mutate_privileged(t, ""),
            lambda r, t: r["long_tokenize"].update(started_at=100, finished_at=101),
            lambda r, t: t["hosts"]["192.168.178.46"]["inspect"].update(started_at=100, finished_at=101),
            lambda r, t: t["hosts"]["192.168.178.46"]["processes"].update(started_at=t["completed_at"] + 1, finished_at=t["completed_at"] + 2),
            lambda r, t: t["final_health"].update(started_at=100, finished_at=101),
            lambda r, t: r["tests"]["vision"]["request"]["messages"][0]["content"][1].update(text="Answer red regardless of image."),
            lambda r, t: r["tests"]["tool"]["request"].update(tools=[]),
            lambda r, t: [row.update(started_at=100, finished_at=101) for row in (
                r["tests"]["direct"], r["tests"]["proxy"], r["tests"]["tool"],
                r["tests"]["vision"], r["tests"]["long_needle"], *r["tests"]["concurrent"])],
            lambda r, t: r["tests"]["long_needle"]["request"]["messages"][0].update(content="Archive row 1837: SABLE-ORCHID-7294"),
            lambda r, t: r["long_tokenize"]["request"].update(model="wrong"),
            lambda r, t: r["tests"]["long_needle"]["response"]["usage"].update(prompt_tokens=1),
            lambda r, t: t["hosts"]["192.168.178.46"]["inspect"].update(returncode=1),
            lambda r, t: t["hosts"]["192.168.178.47"]["kernel"].update(stdout="NVRM: Xid 31"),
            lambda r, t: t["hosts"]["192.168.178.47"]["processes"].update(stdout="PID PPID CMD\n"),
            lambda r, t: t["proxy_models"].update(stdout='[{"model_name":"wrong","api_base":"http://127.0.0.1:8000/v1"}]'),
        ):
            altered_receipt, altered_runtime = copy.deepcopy(receipt), copy.deepcopy(runtime)
            mutate(altered_receipt, altered_runtime)
            with pytest.raises(AssertionError):
                verify_runtime(altered_receipt, altered_runtime)
    finally:
        sys.path.remove(str(evidence))
