#!/usr/bin/env python3
"""Fail-closed static contract for the latest Mia GLM-5.3 recipe."""
from __future__ import annotations

import hashlib
import json
import re
import runpy
import shlex
import subprocess
import sys
import tempfile
import urllib.parse
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml"
MOD = ROOT / "mods/glm-5.3-flash-exl3-upstream-850k"
EVIDENCE = ROOT / "evidence/glm53-exl3-850k-20260913"
SOURCE_REVISION = "f906ee990596486e10ddbe381efa6f0e496f77e3"
IMAGE = (
    "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@"
    "sha256:eecb36e14dc34c92d46827fde7b09f7e0bf27e27c426ece126376c02dea6cd2f"
)
MODEL_REVISION = "024db9f7e9871e8efdf21538ba55af7442be3cd5"
DRAFT_REVISION = "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"


def load_recipe() -> dict:
    assert RECIPE.is_file(), RECIPE
    return yaml.safe_load(RECIPE.read_text())


def test_publication_docs_mark_850k_live_capture_and_preserve_provenance() -> None:
    root_readme = (ROOT / "README.md").read_text()
    evidence_index = (ROOT / "evidence/README.md").read_text()
    evidence_readme = (EVIDENCE / "README.md").read_text()
    superseded_readme = (
        ROOT / "evidence/glm53-exl3-850k-20260909/README.md"
    ).read_text()
    recipe_notes = "\n".join(load_recipe()["metadata"]["notes"])
    assert "850K context (recommended)" in root_readme
    assert "1M context (legacy rollback)" in root_readme
    assert "MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks" in root_readme
    assert SOURCE_REVISION in root_readme
    assert "complete mod tree" in root_readme and "named runtime subset" in root_readme
    assert "850K" in evidence_index and "LIVE_CAPTURE_PASSED" in evidence_index
    gate = evidence_index.split("## Required post-publish round-trip", 1)[1]
    assert "recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml" in gate
    assert "raw.githubusercontent.com/mfellner/sparkrun-recipes/$PUBLISH_SHA" in gate
    assert 'PUBLISHED_TREE="$(mktemp -d)"' in gate
    assert 'git archive "$PUBLISH_SHA" -- "$RECIPE" "$MOD_TREE"' in gate
    assert 'tar -x -C "$PUBLISHED_TREE"' in gate
    assert '"$PUBLISHED_TREE/$RECIPE"' in gate
    assert '"$PUBLISHED_TREE/$MOD_TREE"' in gate
    assert "cmp" in gate and "sparkrun recipe validate" in gate
    validation = root_readme.split("## Validation", 1)[1]
    assert (
        "sparkrun recipe validate recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml"
        in validation
    )
    assert "APPROVED_TREE=" in gate
    assert 'git rev-parse "$PUBLISH_SHA^{tree}"' in gate
    assert 'CONTAINER_NAME="sparkrun_postpublish_glm53_850k"' in gate
    assert gate.count('--container-name "$CONTAINER_NAME"') == 2
    assert "http://127.0.0.1:8000/v1/models" in gate
    assert "http://127.0.0.1:4000/v1/models" in gate
    assert "git status --porcelain" in gate
    assert "sparkrun registry update mfellner" in gate
    assert "@mfellner/glm-5.3-flash-exl3-dflash2-dual-spark-850k" in gate
    assert "vacation-pair2" in gate
    assert "--dry-run --trust" in gate
    assert "trusted namespaced dry-run" in gate
    assert "approved push" in gate and "PENDING" in gate
    assert "Optional destructive live validation" not in gate
    assert "no repository CI workflows" in gate
    for stale in ("qwen3.8", "dual-spark-1m", "spark-pair2"):
        assert stale not in gate
    assert "sparkrun_f906ee990596486e_20260913c411" in evidence_readme
    assert "Acceptance run ID: `f906c41120260913`" in evidence_readme
    assert "Evidence state: **LIVE_CAPTURE_PASSED; PUBLICATION_PENDING**" in evidence_readme
    assert "--run-id" in evidence_readme and "--acceptance-run-id" in evidence_readme
    assert "EXPECTED_ACCEPTANCE_RUN_ID" in evidence_readme
    gates = evidence_readme.split("## Remaining release gates", 1)[1]
    assert re.search(r"two independent\s+fail-closed approvals", gates)
    assert "Exact artifact hashes" in gates
    assert "Fresh live validation: **PASSED**" in root_readme
    assert "Maintainer-approved publication exception" not in root_readme
    assert "despite an independent audit returning" not in root_readme
    assert "**LIVE_CAPTURE_PASSED**" in evidence_index
    assert "SUPERSEDED; HISTORICAL LIVE_CAPTURE_PASSED" in superseded_readme
    assert "../glm53-exl3-850k-20260913/" in superseded_readme
    assert "not runnable release gates" in superseded_readme
    assert "## Remaining release gates" not in superseded_readme
    assert "must not be published" in superseded_readme
    assert "proxy port `4000`" in root_readme
    assert "untrusted" in root_readme.lower()
    assert "proxy port `4000`" in recipe_notes
    assert "untrusted" in recipe_notes.lower()


def test_evidence_readme_runtime_completion_matches_runtime_receipt() -> None:
    runtime = json.loads((EVIDENCE / "runtime.json").read_text())
    readme = (EVIDENCE / "README.md").read_text()
    match = re.search(r"^- Runtime capture completed: `([^`]+)`$", readme, re.MULTILINE)
    assert match is not None
    assert round(datetime.fromisoformat(match.group(1)).timestamp(), 6) == round(
        runtime["completed_at"], 6
    )


def test_verifier_binds_readme_runtime_completion_to_receipt() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    runtime = json.loads((EVIDENCE / "runtime.json").read_text())
    check = verifier["readme_runtime_completion_failures"]
    assert check(EVIDENCE, runtime) == []
    mutated = dict(runtime, completed_at=runtime["completed_at"] + 1)
    assert check(EVIDENCE, mutated) == ["README runtime capture completion"]


def test_root_license_excludes_vendored_upstream_and_model_checkpoints() -> None:
    license_section = (ROOT / "README.md").read_text().split("## License", 1)[1]
    assert "MIT applies only" in license_section
    assert "does not apply" in license_section
    assert "mods/glm-5.3-flash-exl3-upstream-850k/upstream" in license_section
    assert "AGPL-3.0" in license_section
    assert "model checkpoints" in license_section
    assert "ShapleyMCG License 1.0" in license_section
    assert "CC BY-NC-ND 4.0" in license_section


def test_mod_readme_separates_launch_and_full_upstream_gates() -> None:
    readme = (MOD / "README.md").read_text()
    assert "does not run the complete upstream test suite during pre-launch" in readme
    assert "all pytest-compatible upstream checks pass" in readme
    assert "114 passed and 13 subtests passed" in readme
    assert "standalone APC composition gate" in readme
    assert "stale upstream assertion" not in readme
    assert "expected RED" not in readme
    assert "every applicable vendored source/static gate run separately before launch" not in readme


def test_root_pytest_config_excludes_vendored_duplicate_modules() -> None:
    config = (ROOT / "pytest.ini").read_text()
    assert "testpaths = tests" in config
    assert "upstream" not in config


def test_recipe_pins_latest_source_image_and_weights() -> None:
    recipe = load_recipe()
    assert recipe["name"] == "glm-5.3-flash-exl3-dflash2-dual-spark-850k"
    assert recipe["metadata"]["source_revision"] == SOURCE_REVISION
    assert recipe["container"] == IMAGE
    assert recipe["model_revision"] == MODEL_REVISION
    entries = {item["name"]: item["revision"] for item in recipe["distribution_config"]["models"]["entries"]}
    assert entries["Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw"] == MODEL_REVISION
    assert entries["incoai/GLM-5.3-Flash-DFlash2"] == DRAFT_REVISION
    assert f"snapshots/{MODEL_REVISION}" in recipe["command"]
    assert f"snapshots/{DRAFT_REVISION}" in recipe["command"]


def test_refreshed_upstream_gates_reasoning_effort_on_thinking() -> None:
    template = (MOD / "upstream/files/chat_template.jinja").read_text()
    assert (
        "if thinking_enabled and effective_reasoning_effort is not none"
        in template
    )
    assert (
        "if effective_reasoning_effort is not none"
        not in template.replace(
            "if thinking_enabled and effective_reasoning_effort is not none", ""
        )
    )


def test_recipe_matches_latest_safe_upstream_defaults() -> None:
    recipe = load_recipe()
    assert recipe["min_nodes"] == recipe["max_nodes"] == 2
    assert recipe["metadata"]["context_length"] == 850_000
    env = recipe["env"]
    assert str(env["EXL3_FAT_GROUPED"]) == "1"
    assert str(env["EXL3_TEMP_ROWS_FUSED"]) == "32"
    assert str(env["EXL3_FAT_KERNEL"]) == "1"
    assert env["GLM53_INDEXER_WORKSPACE"] == "rightsize"
    assert env["GLM53_ADAPTIVE_K"] == "off"
    assert env["GLM53_DENSE_FP8"] == "off"
    assert recipe["mods"] == ["../mods/glm-5.3-flash-exl3-upstream-850k"]
    assert "/workspace/mods/glm-5.3-flash-exl3-upstream-850k/serve_wrapper.sh" in recipe["command"]


def test_recipe_command_is_an_immutable_non_templated_profile() -> None:
    recipe = load_recipe()
    command = recipe["command"]
    assert recipe["defaults"] == {
        "port": 8000,
        "served_model_name": "GLM-5.3-Flash-EXL3",
        "tensor_parallel": 2,
        "pipeline_parallel": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 850000,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 7168,
    }
    for placeholder in (
        "{served_model_name}",
        "{host}",
        "{port}",
        "{tensor_parallel}",
        "{pipeline_parallel}",
        "{gpu_memory_utilization}",
        "{max_model_len}",
        "{max_num_seqs}",
        "{max_num_batched_tokens}",
        "{dflash_speculative_config}",
        "{limit_mm}",
    ):
        assert placeholder not in command
    for argument in (
        "export GLM53_SERVE_PORT=8000;",
        "export GLM53_SERVED_MODEL=GLM-5.3-Flash-EXL3;",
        "--served-model-name GLM-5.3-Flash-EXL3",
        "--host 0.0.0.0",
        "--port 8000",
        "--tensor-parallel-size 2",
        "--pipeline-parallel-size 1",
        "--gpu-memory-utilization 0.85",
        "--max-model-len 850000",
        "--max-num-seqs 4",
        "--max-num-batched-tokens 7168",
        "--speculative-config '{\"method\":\"dflash\",\"model\":\"/cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/dc77ff1c99eeb2df044ee3d4f0094eb033fee410\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"draft_tensor_parallel_size\":2}'",
        "--limit-mm-per-prompt '{\"image\":4,\"video\":0}'",
    ):
        assert argument in command


def test_verifier_derives_rank_commands_only_from_reviewed_yaml_command() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    contract = verifier["reviewed_recipe_command"](RECIPE)
    assert contract["raw_command"] == load_recipe()["command"]
    for rank in (0, 1):
        runtime = verifier["expected_runtime_command"](rank, contract)
        wrapper = verifier["expected_wrapper_command"](rank, contract)
        assert runtime.startswith("/usr/bin/python3 /usr/local/bin/vllm serve ")
        assert wrapper.startswith(
            "bash /workspace/mods/glm-5.3-flash-exl3-upstream-850k/serve_wrapper.sh "
        )
        assert f"--node-rank {rank}" in runtime
        assert ("--headless" in runtime) is (rank == 1)
        assert '--limit-mm-per-prompt {"image":4,"video":0}' in runtime


def test_reviewed_command_parser_rejects_executable_ambiguity(tmp_path: Path) -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    recipe = load_recipe()
    mutations = (
        recipe["command"] + "; touch /tmp/forged\n",
        recipe["command"].replace("exec bash ", "exec bash -c 'true' && bash ", 1),
        recipe["command"].replace("--port 8000", "--port 8000 --port 8001", 1),
    )
    for index, command in enumerate(mutations):
        changed = dict(recipe, command=command)
        path = tmp_path / f"ambiguous-{index}.yaml"
        path.write_text(yaml.safe_dump(changed, sort_keys=False))
        try:
            verifier["reviewed_recipe_command"](path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted ambiguous executable mutation {index}")


def test_recipe_denies_arbitrary_remote_media_fetches() -> None:
    recipe = load_recipe()
    command = recipe["command"]
    assert "--allowed-media-domains media.invalid" in command
    assert "--allowed-media-domains '*'" not in command
    assert "--allowed-media-domains \"*\"" not in command
    notes = "\n".join(recipe["metadata"]["notes"])
    assert "data URLs" in notes
    assert "media.invalid" in notes
    assert "unauthenticated" in notes
    assert "trusted firewalled private network" in notes


def test_roce_gid_helper_fails_closed_per_hca() -> None:
    helper_path = MOD / "verify_roce_gid.py"
    assert helper_path.is_file()
    verify_ns = runpy.run_path(str(helper_path))
    verify_hcas = verify_ns["verify_hcas"]
    capture_hcas = verify_ns["capture_hcas"]

    def install(root: Path, hca: str, gid: str, gid_type: str, ndev: str) -> None:
        base = root / "class/infiniband" / hca / "ports/1"
        for rel, value in (
            (f"gids/3", gid),
            (f"gid_attrs/types/3", gid_type),
            (f"gid_attrs/ndevs/3", ndev),
        ):
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value + "\n")

    with tempfile.TemporaryDirectory(prefix="glm53-roce-") as tmp:
        root = Path(tmp)
        install(root, "rocep1s0f1", "0000:0000:0000:0000:0000:ffff:c0a8:0348", "RoCE v2", "enp1s0f1np1")
        install(root, "roceP2p1s0f1", "0000:0000:0000:0000:0000:ffff:c0a8:0248", "RoCE v2", "enP2p1s0f1np1")
        addresses = {
            "enp1s0f1np1": ["192.168.3.72"],
            "enP2p1s0f1np1": ["192.168.2.72"],
        }
        lookup = lambda ndev: addresses.get(ndev, [])
        assert verify_hcas(root, 3, ["rocep1s0f1", "roceP2p1s0f1"], lookup) == []
        assert capture_hcas(
            root, 3, ["rocep1s0f1", "roceP2p1s0f1"], lookup
        ) == [
            {
                "hca": "rocep1s0f1",
                "gid_index": 3,
                "gid": "::ffff:192.168.3.72",
                "gid_type": "RoCE v2",
                "netdev": "enp1s0f1np1",
                "ipv4": "192.168.3.72",
            },
            {
                "hca": "roceP2p1s0f1",
                "gid_index": 3,
                "gid": "::ffff:192.168.2.72",
                "gid_type": "RoCE v2",
                "netdev": "enP2p1s0f1np1",
                "ipv4": "192.168.2.72",
            },
        ]

        (root / "class/infiniband/rocep1s0f1/ports/1/gid_attrs/types/3").write_text("RoCE v1\n")
        assert any("RoCE v2" in failure for failure in verify_hcas(root, 3, ["rocep1s0f1"], lookup))
        (root / "class/infiniband/rocep1s0f1/ports/1/gid_attrs/types/3").write_text("RoCE v2\n")

        addresses["enp1s0f1np1"] = ["192.168.3.99"]
        assert any("does not map" in failure for failure in verify_hcas(root, 3, ["rocep1s0f1"], lookup))
        addresses["enp1s0f1np1"] = ["192.168.3.72"]

        (root / "class/infiniband/rocep1s0f1/ports/1/gids/3").write_text("2001:db8::c0a8:0348\n")
        assert any("IPv4-mapped" in failure for failure in verify_hcas(root, 3, ["rocep1s0f1"], lookup))
        (root / "class/infiniband/rocep1s0f1/ports/1/gids/3").write_text("0000:0000:0000:0000:0000:ffff:c0a8:0348\n")
        assert any("GID index" in failure for failure in verify_hcas(root, -1, ["rocep1s0f1"], lookup))

        (root / "class/infiniband/roceP2p1s0f1/ports/1/gid_attrs/ndevs/3").unlink()
        assert any("ndev" in failure for failure in verify_hcas(root, 3, ["rocep1s0f1", "roceP2p1s0f1"], lookup))

        install(root, "roceP2p1s0f1", "0000:0000:0000:0000:0000:ffff:c0a8:0248", "RoCE v1", "enP2p1s0f1np1")
        failures = verify_hcas(root, 3, ["rocep1s0f1", "roceP2p1s0f1"], lookup)
        assert any("roceP2p1s0f1" in failure and "RoCE v2" in failure for failure in failures)


def test_run_invokes_per_hca_rocev2_ipv4_gate() -> None:
    helper = (MOD / "verify_roce_gid.py").read_text()
    assert "SIOCGIFADDR" in helper
    assert "subprocess" not in helper
    assert '["ip"' not in helper
    run = (MOD / "run.sh").read_text()
    assert "verify_roce_gid.py" in run
    assert '"${hcas[@]}"' in run
    assert "tr -d ':0'" not in run


def test_runtime_mod_applies_latest_pure_python_overlays_and_checks_e3() -> None:
    run = (MOD / "run.sh").read_text()
    assert SOURCE_REVISION in run
    expected_order = [
        "python3 upstream/overlay/patch_hybrid_prefix_hit.py",
        "python3 upstream/overlay/patch_apc_per_group_retention.py",
        "python3 upstream/overlay/patch_spinwait.py",
        "python3 upstream/overlay/patch_adaptive_k.py",
        "python3 patch_e3_execution_marker.py",
        "python3 upstream/overlay/patch_dense_fp8.py",
        "python3 upstream/overlay/patch_indexer_workspace.py",
        "python3 patch_ablit.py",
    ]
    positions = [run.index(item) for item in expected_order]
    assert positions == sorted(positions)
    assert "upstream/overlay/exl3.py" in run
    for symbol in ("exl3_fat_moe_gateup", "exl3_fat_moe_down", "exl3_fat_moe_gather"):
        assert symbol in run
    assert (MOD / "patch_e3_execution_marker.py").is_file()
    assert (MOD / "test_e3_execution_marker.py").is_file()
    assert (MOD / "upstream" / "overlay" / "patch_adaptive_k.py").is_file()
    assert (MOD / "upstream" / "overlay" / "patch_dense_fp8.py").is_file()
    assert (MOD / "upstream" / "overlay" / "exl3_fat_moe.cu").is_file()
    assert (MOD / "upstream" / "LICENSE").is_file()


def test_e3_execution_marker_is_runtime_bound_on_both_ranks() -> None:
    runtime_verifier = (MOD / "verify_runtime_patch_state.py").read_text()
    capture = (EVIDENCE / "capture_runtime.py").read_text()
    verifier = (EVIDENCE / "verify.py").read_text()
    marker = "[glm53-e3-executed]"
    for text in (runtime_verifier, capture, verifier):
        assert marker in text
    assert "grouped_calls" in verifier and "fat_expert_runs" in verifier
    assert "E3 runtime execution marker" in verifier


def test_runtime_mod_verifies_complete_patched_state_after_application() -> None:
    verifier = MOD / "verify_runtime_patch_state.py"
    assert verifier.is_file()
    run = (MOD / "run.sh").read_text()
    verifier_command = "python3 verify_runtime_patch_state.py"
    assert run.count(verifier_command) == 1
    assert run.index(verifier_command) > run.index("python3 patch_ablit.py")
    assert run.index(verifier_command) < run.index("import torch, exllamav3_ext")
    result = subprocess.run(
        [sys.executable, str(verifier), "--self-test"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime patch verifier self-test: PASS" in result.stdout
    assert "contracts=12" in result.stdout
    assert "embedded_forbidden=10" in result.stdout
    mutations = int(result.stdout.rsplit("mutations=", 1)[1].split()[0])
    assert mutations >= 40


def test_runtime_patch_verifier_targets_the_actual_upstream_files() -> None:
    verifier = (MOD / "verify_runtime_patch_state.py").read_text()
    assert 'site / "v1/attention/backends/mla/indexer.py"' in verifier
    assert 'site / "v1/structured_output/__init__.py"' in verifier
    assert 'site / "v1/attention/backends/mla/sparse.py"' not in verifier
    assert 'site / "v1/structured_output/manager.py"' not in verifier


def test_runtime_patch_verifier_counts_embedded_forbidden_occurrences() -> None:
    ns = runpy.run_path(str(MOD / "verify_runtime_patch_state.py"))
    fragment_failures = ns["fragment_failures"]
    adaptive = runpy.run_path(str(MOD / "upstream/overlay/patch_adaptive_k.py"))
    required = [
        ns["load_script"](MOD / "upstream/overlay/patch_scheduler_decode_floor.py")["HELPER"],
        ns["load_script"](MOD / "upstream/overlay/patch_scheduler_decode_floor.py")["RUNNING_NEW"],
        ns["load_script"](MOD / "upstream/overlay/patch_scheduler_decode_floor.py")["WAITING_NEW"],
        adaptive["SCHED_HELPER"],
        adaptive["OBS_NEW"],
        adaptive["UPD_NEW"],
        adaptive["SCHED_K_NEW"],
    ]
    forbidden = [
        adaptive["OBS_OLD"],
        adaptive["UPD_OLD"],
        adaptive["SCHED_K_OLD"],
    ]
    clean = "\n".join(required)
    assert fragment_failures("scheduler", clean, required, forbidden) == []
    for forbidden_index in (0, 2):
        mixed = clean + "\n" + forbidden[forbidden_index]
        failures = fragment_failures("scheduler", mixed, required, forbidden)
        assert any(
            f"forbidden fragment {forbidden_index}" in failure for failure in failures
        ), (forbidden_index, failures)


def test_stop_policy_arms_from_authoritative_state_with_suffix_fallback() -> None:
    patch_ns = runpy.run_path(str(MOD / "patch_suppress_stops_multitoken.py"))
    new = patch_ns["NEW"]
    assert "reasoning_ended is False" in new
    assert "reasoning_ended is True" in new
    assert "reasoning_ended is None" in new
    assert "ptids[-1] == think_id" in new

    source = """class IncrementalDetokenizer:
    @staticmethod
    def _suppress_stops_enabled():
        return True
    @staticmethod
    def _reasoning_stop_markers():
        return '<think>', '</think>'
    @staticmethod
    def arm(detok, tokenizer, request):
        try:
            stop = getattr(detok, 'stop', None)
            ptids = getattr(request, 'prompt_token_ids', None)
            if not stop:
                return
""" + new + """        except Exception:
            return
"""
    namespace: dict = {}
    exec(compile(source, "<stop-arm-contract>", "exec"), namespace)
    cls = namespace["IncrementalDetokenizer"]

    class Tokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<think>"
            return 999

    class Detok:
        stop = ["Question:"]
        _reasoning_stop_guard = False
        _reasoning_end_str = ""

    for state, prompt, expected in (
        (False, [1, 2], True),
        (True, [1, 999], False),
        (None, [1, 999], True),
        (None, [1, 2], False),
    ):
        request = type("Request", (), {"reasoning_ended": state, "prompt_token_ids": prompt})()
        detok = Detok()
        cls.arm(detok, Tokenizer(), request)
        assert detok._reasoning_stop_guard is expected, (state, prompt)


def test_stop_policy_checks_only_after_reasoning_closes_or_when_unarmed() -> None:
    patch_ns = runpy.run_path(str(MOD / "patch_suppress_stops_multitoken.py"))
    check = patch_ns["CHECK_NEW"]
    assert "_suppress_stops_enabled" not in check
    source = """class Probe:
    def check(self):
        return (
            self.stop
            and self.num_output_tokens() > self.min_tokens
""" + check + """        )
"""
    namespace: dict = {}
    exec(compile(source, "<stop-check-contract>", "exec"), namespace)
    probe = namespace["Probe"]()
    probe.stop = ["Question:"]
    probe.num_output_tokens = lambda: 2
    probe.min_tokens = 0
    for guarded, closed, expected in (
        (True, False, False),
        (True, True, True),
        (False, False, True),
    ):
        probe._reasoning_stop_guard = guarded
        probe._reasoning_closed = closed
        assert probe.check() is expected, (guarded, closed)


def test_stop_policy_repairs_preexisting_global_suppression() -> None:
    patch_ns = runpy.run_path(str(MOD / "patch_suppress_stops_multitoken.py"))
    bad = """class IncrementalDetokenizer:
    @staticmethod
    def probe(detok, tokenizer, request):
        try:
""" + patch_ns["BAD_NEW"] + """        except Exception:
            return
    def check(self):
        return (
            self.stop
            and self.num_output_tokens() > self.min_tokens
""" + patch_ns["CHECK_BAD"] + """        )
"""
    repaired, status = patch_ns["apply_text"](bad)
    assert status == "repaired"
    assert patch_ns["NEW"] in repaired
    assert patch_ns["CHECK_NEW"] in repaired
    assert patch_ns["BAD_NEW"] not in repaired
    assert patch_ns["CHECK_BAD"] not in repaired
    same, status = patch_ns["apply_text"](repaired)
    assert status == "skipped"
    assert same == repaired


def test_archival_boot_candidate_is_not_in_the_runtime_call_graph() -> None:
    runtime_files = (
        RECIPE,
        MOD / "run.sh",
        MOD / "serve_wrapper.sh",
        MOD / "postready_gate.sh",
    )
    for path in runtime_files:
        assert "boot_candidate.sh" not in path.read_text(), path
    readme = (MOD / "README.md").read_text()
    assert "boot_candidate.sh" in readme
    assert "archival" in readme
    assert "not invoked" in readme


def test_postready_gate_warms_exact_acceptance_concurrency_shape() -> None:
    gate = (MOD / "postready_gate.sh").read_text()
    assert "threading.Barrier(4)" in gate
    assert "ThreadPoolExecutor(max_workers=4)" in gate
    assert '"max_tokens":32' in gate
    assert "GLM53_GATE_C4_" in gate
    assert 'filler="alpha "*55000' in gate
    assert 'needle="NEEDLE_GL53_842917"' in gate
    assert "timeout=1800" in gate
    readiness_marker = gate.index('date --iso-8601=seconds > "$OK"')
    assert gate.index("ThreadPoolExecutor(max_workers=4)") < readiness_marker
    assert gate.index('filler="alpha "*55000') < readiness_marker
    capture = (EVIDENCE / "capture_runtime.py").read_text()
    verifier = (EVIDENCE / "verify.py").read_text()
    assert "kernel_readiness_to_acceptance" in capture
    assert "kernel_readiness_to_acceptance" in verifier
    assert "NV_ERR_NO_MEMORY" in verifier
    assert "capture was incomplete" in capture
    assert 'f"{host} {kernel_name} stderr"' in verifier


def test_acceptance_binds_remote_error_scoped_stops_and_telemetry_signal() -> None:
    acceptance = (EVIDENCE / "acceptance.py").read_text()
    verifier = (EVIDENCE / "verify.py").read_text()
    collector = EVIDENCE / "capture_load_telemetry.py"
    assert collector.is_file()
    assert "--telemetry-signal" in acceptance
    assert "REMOTE_MEDIA_ERROR" in acceptance
    assert '"type": "BadRequestError"' in verifier
    assert '"param": None' in verifier
    assert "direct_reasoning_open_stop" in acceptance and "direct_reasoning_open_stop" in verifier
    assert "direct_thinking_disabled_stop" in acceptance and "direct_thinking_disabled_stop" in verifier
    assert "telemetry direct-load overlap" in verifier
    assert '"capture_load_telemetry.py"' in verifier


def test_reasoning_open_stop_request_uses_non_repeated_semantic_stop() -> None:
    expected = {
        "model": "GLM-5.3-Flash-EXL3",
        "messages": [{
            "role": "user",
            "content": (
                "Compute 12 times 12. If the result is 144, answer with exactly "
                "GLM53_REASONING_STOP_OK and nothing else."
            ),
        }],
        "stop": ["144"],
        "temperature": 0,
        "max_tokens": 2300,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    acceptance_ns = runpy.run_path(str(EVIDENCE / "acceptance.py"))
    verifier_ns = runpy.run_path(str(EVIDENCE / "verify.py"))
    assert acceptance_ns["reasoning_open_stop_request"]() == expected
    assert verifier_ns["reasoning_open_stop_request"]() == expected


def test_reasoning_stop_requires_exact_content_and_stop_finish_reason() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    choice = {
        "message": {
            "reasoning": "computed 144 before closing reasoning",
            "content": "GLM53_REASONING_STOP_OK",
        },
        "finish_reason": "stop",
        "stop_reason": 154827,
    }
    assert verifier["reasoning_stop_failures"](choice) == []
    prefixed = json.loads(json.dumps(choice))
    prefixed["message"]["content"] = "prefix GLM53_REASONING_STOP_OK"
    assert "reasoning-open exact final answer" in verifier["reasoning_stop_failures"](
        prefixed
    )
    length = json.loads(json.dumps(choice))
    length["finish_reason"] = "length"
    assert "reasoning-open finish reason" in verifier["reasoning_stop_failures"](length)
    forged_stop = json.loads(json.dumps(choice))
    forged_stop["stop_reason"] = 999999
    assert "reasoning-open exact stop reason" in verifier["reasoning_stop_failures"](
        forged_stop
    )
    spaced = json.loads(json.dumps(choice))
    spaced["message"]["content"] = " GLM53_REASONING_STOP_OK "
    assert "reasoning-open exact final answer" in verifier["reasoning_stop_failures"](
        spaced
    )
    acceptance = (EVIDENCE / "acceptance.py").read_text()
    assert 'reasoning_content == "GLM53_REASONING_STOP_OK"' in acceptance
    assert 'reasoning_choice.get("finish_reason") == "stop"' in acceptance
    controls = (EVIDENCE / "test_negative_controls.py").read_text()
    assert "reasoning-open final answer prefixed" in controls
    assert "reasoning-open finish reason changed to length" in controls
    assert "reasoning-open stop reason forged with refreshed binding and manifest" in controls
    assert "thinking-disabled finish reason changed to length with refreshed binding and manifest" in controls


def test_thinking_disabled_stop_requires_exact_finish_metadata() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    choice = {
        "message": {"reasoning": None, "content": "BEFORE "},
        "finish_reason": "stop",
        "stop_reason": "Question:",
    }
    assert verifier["thinking_disabled_stop_failures"](choice) == []
    length = json.loads(json.dumps(choice))
    length["finish_reason"] = "length"
    assert "thinking-disabled finish reason" in verifier[
        "thinking_disabled_stop_failures"
    ](length)


def test_evidence_verifier_has_exact_command_and_final_artifact_bindings() -> None:
    verifier = (EVIDENCE / "verify.py").read_text()
    verifier_ns = runpy.run_path(str(EVIDENCE / "verify.py"))
    assert "expected_runtime_overlay_argv" in verifier
    assert "expected_postready_argv" in verifier
    assert "FINAL_ARTIFACT_SHA256" in verifier
    assert set(verifier_ns["FINAL_ARTIFACT_SHA256"]) == {
        "acceptance.py",
        "capture_runtime.py",
        "capture_load_telemetry.py",
        "command_contract.py",
        "acceptance.json",
        "runtime.json",
        "load-telemetry.log",
        "proxy-models.json",
        "proxy-start.log",
        "launch-epoch.txt",
        "launch-receipt.json",
        "launch.log",
        "vision-quadrants.png",
        "vision-ocr.png",
        "video-tiny.gif",
        "README.md",
        "static-validation.log",
    }
    bindings = verifier_ns["FINAL_ARTIFACT_SHA256"]
    values = set(bindings.values())
    assert values == {"PENDING_RECAPTURE"} or all(
        re.fullmatch(r"[0-9a-f]{64}", value) for value in values
    )
    if values != {"PENDING_RECAPTURE"}:
        for rel, expected in bindings.items():
            assert hashlib.sha256((EVIDENCE / rel).read_bytes()).hexdigest() == expected
    assert "verify.py" not in verifier_ns["FINAL_ARTIFACT_SHA256"]
    assert "reviewed git tree and this verifier are the external trust root" in verifier
    assert verifier_ns["CLUSTER"] == "sparkrun_f906ee990596486e_20260913c411"
    assert "artifact_binding_failures" in verifier
    controls = (EVIDENCE / "test_negative_controls.py").read_text()
    assert 'names = tuple(verify["FINAL_ARTIFACT_SHA256"])' in controls


def test_acceptance_and_runtime_metadata_fail_closed() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    run_id = "0123456789abcdef"
    launch_epoch = 1_788_988_887
    acceptance = {
        "schema": 1,
        "process_role": "exact_final",
        "passed": True,
        "run_id": run_id,
    }
    runtime = {
        "schema": 1,
        "process_role": "exact_final",
        "producer": "capture_runtime.py",
        "acceptance_run_id": run_id,
        "capture_argv": verifier["expected_capture_argv"](launch_epoch, run_id),
    }
    validate = verifier["top_level_metadata_failures"]
    assert validate(acceptance, runtime, launch_epoch, run_id) == []
    cases = (
        ("acceptance schema", acceptance, "schema", 2, "acceptance schema/process/verdict"),
        ("acceptance boolean schema", acceptance, "schema", True, "acceptance schema/process/verdict"),
        ("acceptance role", acceptance, "process_role", "draft", "acceptance schema/process/verdict"),
        ("acceptance verdict", acceptance, "passed", False, "acceptance schema/process/verdict"),
        ("acceptance run", acceptance, "run_id", "ffffffffffffffff", "acceptance exact run ID"),
        ("runtime schema", runtime, "schema", 2, "runtime schema/process/producer"),
        ("runtime boolean schema", runtime, "schema", True, "runtime schema/process/producer"),
        ("runtime role", runtime, "process_role", "draft", "runtime schema/process/producer"),
        ("runtime producer", runtime, "producer", "other.py", "runtime schema/process/producer"),
        ("runtime run", runtime, "acceptance_run_id", "ffffffffffffffff", "runtime/acceptance run ID join"),
        ("runtime argv", runtime, "capture_argv", ["capture_runtime.py"], "runtime capture argv"),
    )
    for name, original, key, replacement, expected in cases:
        mutated_acceptance = json.loads(json.dumps(acceptance))
        mutated_runtime = json.loads(json.dumps(runtime))
        target = mutated_acceptance if original is acceptance else mutated_runtime
        target[key] = replacement
        assert expected in validate(
            mutated_acceptance, mutated_runtime, launch_epoch, run_id
        ), name
    assert "acceptance run ID pending recapture" in validate(
        acceptance, runtime, launch_epoch, "PENDING_RECAPTURE"
    )

    acceptance_source = (EVIDENCE / "acceptance.py").read_text()
    runtime_source = (EVIDENCE / "capture_runtime.py").read_text()
    assert 'ap.add_argument("--run-id", required=True)' in acceptance_source
    assert 'parser.add_argument("--acceptance-run-id", required=True)' in runtime_source
    assert '"process_role": "exact_final"' in runtime_source
    assert '"producer": "capture_runtime.py"' in runtime_source


def test_honest_refresh_controls_cover_review_blockers() -> None:
    controls = (EVIDENCE / "test_negative_controls.py").read_text()
    for control in (
        "wrapper port mutated with refreshed binding",
        "wrapper flag removed with refreshed binding",
        "worker rank mutated with refreshed binding",
        "vLLM port mutated with refreshed binding",
        "vLLM flag removed with refreshed binding",
        "vLLM worker rank mutated with refreshed binding",
        "acceptance run ID substituted with refreshed binding",
        "acceptance schema substituted with refreshed binding",
        "acceptance process role substituted with refreshed binding",
        "acceptance passed verdict substituted with refreshed binding",
        "runtime schema substituted with refreshed binding",
        "runtime process role substituted with refreshed binding",
        "runtime producer substituted with refreshed binding",
        "runtime capture argv substituted with refreshed binding",
        "response fingerprint substituted with refreshed binding",
        "response created timestamp zeroed with refreshed binding",
        "worker negative listener receipt {mode} with refreshed binding",
        "head vLLM PID substituted with refreshed binding",
        "worker vLLM PID substituted with refreshed binding",
        "head wrapper PID substituted with refreshed binding",
        "worker wrapper PID substituted with refreshed binding",
        "recipe video limit changed 0 to 1 with all bindings refreshed",
        "recipe executable flag added with all bindings refreshed",
        "recipe executable token added with all bindings refreshed",
        "recipe shell statement added with all bindings refreshed",
        "{route} video rejection changed to success",
        "{route} video request substituted self-consistently",
        "video fixture and both inline rejection requests replaced self-consistently",
        "unknown worker process added with refreshed binding",
        "extra worker listener process added with refreshed binding",
    ):
        assert control in controls


def test_successful_chat_receipts_bind_fingerprint_and_created_interval() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    row = {
        "started_at": 100.75,
        "completed_at": 101.25,
        "response": {
            "system_fingerprint": verifier["EXPECTED_SYSTEM_FINGERPRINT"],
            "created": 100,
        },
    }
    validate = verifier["response_runtime_failures"]
    assert validate("direct_exact", row) == []
    wrong_fingerprint = json.loads(json.dumps(row))
    wrong_fingerprint["response"]["system_fingerprint"] = "forged-runtime"
    assert "direct_exact system fingerprint" in validate("direct_exact", wrong_fingerprint)
    zero_created = json.loads(json.dumps(row))
    zero_created["response"]["created"] = 0
    assert "direct_exact response created interval" in validate("direct_exact", zero_created)
    boolean_created = json.loads(json.dumps(row))
    boolean_created["response"]["created"] = True
    assert "direct_exact response created interval" in validate("direct_exact", boolean_created)
    crossed_second = json.loads(json.dumps(row))
    crossed_second["response"]["created"] = 101
    assert validate("direct_exact", crossed_second) == []


def test_acceptance_records_sustained_load_for_telemetry() -> None:
    acceptance = (EVIDENCE / "acceptance.py").read_text()
    verifier = (EVIDENCE / "verify.py").read_text()
    collector = (EVIDENCE / "capture_load_telemetry.py").read_text()
    assert "direct_telemetry_load" in acceptance
    assert '"ignore_eos": True' in acceptance
    assert 'write_signal(args.telemetry_signal, "acceptance_completed"' in acceptance
    assert acceptance.count("wait_for_collector(args.telemetry_signal)") == 1
    assert acceptance.index("wait_for_collector(args.telemetry_signal)") < acceptance.index('"started_at": time.time()')
    assert 'wait_state(args.signal, "acceptance_completed", args.timeout)' in collector
    assert "direct_telemetry_load" in verifier
    assert "telemetry load completion tokens" in verifier


def test_acceptance_proves_remote_media_is_rejected_on_direct_and_proxy_routes() -> None:
    acceptance = (EVIDENCE / "acceptance.py").read_text()
    verifier_ns = runpy.run_path(str(EVIDENCE / "verify.py"))
    assert "direct_remote_media_rejected" in acceptance
    assert "proxy_remote_media_rejected" in acceptance
    assert "http://127.0.0.1:8000/v1/models" in acceptance
    assert {
        "direct_remote_media_rejected",
        "proxy_remote_media_rejected",
    }.issubset(verifier_ns["EXPECTED_CHECKS"])
    assert verifier_ns["PROXY_REMOTE_MEDIA_ERROR"] == {
        "error": {
            "code": "400",
            "message": (
                "litellm.BadRequestError: OpenAIException - The URL must be from one of the allowed domains: "
                "['media.invalid']. Input URL domain: 127.0.0.1. Received Model Group=GLM-5.3-Flash-EXL3\n"
                "Available Model Group Fallbacks=None"
            ),
            "param": None,
            "type": None,
        }
    }
    negative_controls = (EVIDENCE / "test_negative_controls.py").read_text()
    assert "proxy remote media rejection changed to success" in negative_controls
    assert "proxy remote media allowlist error substituted" in negative_controls
    for key in ("direct_remote_media_rejected", "proxy_remote_media_rejected"):
        route = "http://127.0.0.1:4000" if key.startswith("proxy") else "http://127.0.0.1:8000"
        body = verifier_ns["remote_media_request"]()
        row = {
            "passed": True,
            "requested_url": route + "/v1/chat/completions",
            "effective_url": route + "/v1/chat/completions",
            "path": "/v1/chat/completions",
            "http": 400,
            "request": body,
            "request_sha256": verifier_ns["request_sha"](body),
            "response": (
                verifier_ns["PROXY_REMOTE_MEDIA_ERROR"]
                if key.startswith("proxy")
                else verifier_ns["REMOTE_MEDIA_ERROR"]
            ),
        }
        assert verifier_ns["remote_media_failures"](key, row) == []


def test_evidence_verifier_requires_exact_metadata_defaults() -> None:
    verifier = (EVIDENCE / "verify.py").read_text()
    assert "IMMUTABLE_DEFAULTS" in verifier
    assert 'require(raw_recipe.get("defaults") == IMMUTABLE_DEFAULTS' in verifier


def test_capture_redacts_credentials_without_losing_required_runtime_fields() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    adversarial = {
        "AWS_ACCESS_KEY_ID": "sentinel-aws-value",
        "PGPASSWORD": "sentinel-pg-value",
        "GITHUB_PAT": "sentinel-github-value",
        "COOKIE": "sentinel-cookie-value",
        "PASSPHRASE": "sentinel-passphrase-value",
        "private_key": "sentinel-private-key-value",
        "CUSTOM_RUNTIME_VALUE": "sentinel-generic-credential-value",
    }
    env = ["NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1"] + [
        f"{name}={value}" for name, value in adversarial.items()
    ]
    sample = {
        "inspect": {"Config": {"Env": env, "Image": IMAGE}},
        "docker_stdout": json.dumps([{"Config": {"Env": env}}]),
        "logs": "\n".join(env) + "\n",
        "command": "litellm --port 4000 --unknown-option sentinel-command-value",
    }
    redacted = capture["sanitize_evidence"](sample)
    serialized = json.dumps(redacted)
    for value in (*adversarial.values(), "sentinel-command-value"):
        assert value not in serialized
    assert serialized.count("[REDACTED]") >= 22
    docker_stdout = json.loads(redacted["docker_stdout"])
    assert docker_stdout[0]["Config"]["Env"] == [
        "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1",
        *[f"{name}=[REDACTED]" for name in adversarial],
    ]
    assert "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1" in serialized
    assert IMAGE in serialized
    capture_source = (EVIDENCE / "capture_runtime.py").read_text()
    assert "sanitize_evidence(record)" in capture_source
    assert "SAFE_ENV_NAMES" in capture_source
    env_command = capture["runtime_env_command"]()
    assert "grep" not in env_command and "env |" not in env_command
    for name in capture["SAFE_ENV_NAMES"]:
        assert env_command.count(name) == 1

    marker = "synthetic-sensitive-marker"
    expected_runtime = runpy.run_path(str(EVIDENCE / "verify.py"))[
        "expected_runtime_command"
    ](0)
    reviewer_examples = {
        "api_key_argv": ["tool", "--api-key", marker],
        "header_argv": ["curl", "-H", f"Authorization: Bearer {marker}"],
        "userinfo_argv": ["curl", f"https://user:{marker}@example.invalid/path"],
        "credential_vllm_command": expected_runtime + f" --api-key {marker}",
        "credential_safe_env": f"NCCL_IB_HCA=https://user:{marker}@example.invalid",
        "common_credential_json": json.dumps(
            {
                "client_secret": marker,
                "accessKeyId": marker,
                "sessionToken": marker,
                "privateKey": marker,
            }
        ),
        "standard_env_mapping": {"PGPASSWORD": marker},
        "prefixed_and_suffixed_keys": {
            "api_key_backup": marker,
            "secret_value": marker,
        },
        "unknown_assignment": f"UNRECOGNIZED_RUNTIME_VALUE={marker}",
        "nested": {"probe_argv": ["probe", "--header", f"X-Api-Key: {marker}"]},
    }
    adversarial_redacted = capture["sanitize_evidence"](reviewer_examples)
    assert marker not in json.dumps(adversarial_redacted)
    assert adversarial_redacted["credential_vllm_command"] == "[REDACTED]"
    assert adversarial_redacted["credential_safe_env"] == "[REDACTED]"
    assert adversarial_redacted["common_credential_json"] == "[REDACTED]"

    canonical = {
        "path": "/v1/models",
        "requested_url": "http://127.0.0.1:8000/v1/models",
        "argv": ["ssh", "-o", "BatchMode=yes", "192.168.178.47", "bash", "-lc", "pid=123"],
        "probe_argv": ["python3", "-c", capture["LISTENER_PROBE"], "4000"],
        "command": load_recipe()["command"],
        "ulimits": ["memlock=-1:-1", "stack=67108864:67108864", "nofile=65535:65535"],
        "stdout": (
            "pid=674\n"
            "path=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2\n"
            "sha256=fc7ea66334edbc934aa25959b9907dbb2b91a1d2485beff18839afc45cbc08d0\n"
            "package_version=2.30.7\n"
            "symbols=exl3_moe,exl3_fat_gemm,exl3_fat_gemm_scatter,exl3_fat_moe_gateup,exl3_fat_moe_down,exl3_fat_moe_gather\n"
            "oom_kill=0\n"
            "label=disable\n"
            "AWS_ACCESS_KEY_ID=credential-like-value\n"
        ),
    }
    canonical_redacted = capture["sanitize_evidence"](canonical)
    assert canonical_redacted["path"] == canonical["path"]
    assert canonical_redacted["requested_url"] == canonical["requested_url"]
    assert canonical_redacted["argv"] == canonical["argv"]
    assert canonical_redacted["probe_argv"] == canonical["probe_argv"]
    assert canonical_redacted["command"] == canonical["command"]
    assert canonical_redacted["ulimits"] == canonical["ulimits"]
    for safe_line in canonical["stdout"].splitlines()[:-1]:
        assert safe_line in canonical_redacted["stdout"]
    assert "AWS_ACCESS_KEY_ID=[REDACTED]" in canonical_redacted["stdout"]


def test_capture_sanitizer_rejects_reviewer_credential_matrix() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    marker = "reviewer-sensitive-marker"
    cases = (
        "https://example.invalid/path?token=" + marker,
        "https://example.invalid/path?api_key=" + marker,
        "https://example.invalid/path?access_token=" + marker,
        "https://example.invalid/path?auth_token=" + marker,
        "https://example.invalid/path?refresh_token=" + marker,
        {"X-Api-Key": marker},
        {"X-Auth-Token": marker},
        {"outer": {"x_api_key": marker}},
        {"outer": [{"x-auth-token": marker}]},
        json.dumps({"headers": {"X-Api-Key": marker}}),
        json.dumps({"url": "https://example.invalid/?access_token=" + marker}),
        "X-Api-Key: " + marker,
        "X-Auth-Token: " + marker,
        {"probe_argv": ["curl", "--header", "X-Api-Key: " + marker]},
        {"probe_argv": ["curl", "--http-header=X-Auth-Token: " + marker]},
        [{"apiToken": marker}],
        "https://example.invalid/?safe=1&session_token=" + marker + "&other=2",
    )
    assert len(cases) == 17
    for index, case in enumerate(cases):
        sanitized = capture["sanitize_evidence"](case)
        assert marker not in json.dumps(sanitized), index
        assert "[REDACTED]" in json.dumps(sanitized), index


def test_capture_sanitizer_classifies_percent_decoded_uri_query_names() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    marker = "encoded-query-secret"
    cases = {
        "https://example.invalid/?%61pi_key=" + marker: "%61pi_key=[REDACTED]",
        "https://example.invalid/?%2561pi_key=" + marker: "%2561pi_key=[REDACTED]",
        "https://example.invalid/?api_key%3D" + marker: "api_key=[REDACTED]",
        "https://example.invalid/?x%2Dauth%2Dtoken=" + marker: "x%2Dauth%2Dtoken=[REDACTED]",
        "https://example.invalid/?safe%5Fname=" + marker: "safe%5Fname=" + marker,
    }
    for value, expected in cases.items():
        sanitized = capture["sanitize_evidence"](value)
        assert expected in sanitized
        if "safe%5Fname" not in value:
            assert marker not in sanitized


def test_capture_sanitizer_recurses_through_bounded_serialized_json_strings() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    marker = "multiply-serialized-secret"

    for header, layers in (("X-Api-Key", 2), ("X-Auth-Token", 3)):
        value = json.dumps({"headers": {header: marker}, "safe": "preserved"})
        for _ in range(layers - 1):
            value = json.dumps(value)
        sanitized = capture["sanitize_evidence"](value)
        assert marker not in sanitized
        decoded = sanitized
        for _ in range(layers):
            decoded = json.loads(decoded)
        assert decoded == {"headers": {header: "[REDACTED]"}, "safe": "preserved"}

    too_deep = json.dumps({"X-Api-Key": marker})
    for _ in range(capture["JSON_SANITIZE_MAX_DEPTH"] + 1):
        too_deep = json.dumps(too_deep)
    assert capture["sanitize_evidence"](too_deep) == capture["REDACTED"]

    oversized = json.dumps({"X-Api-Key": marker, "padding": "x" * 64})
    assert capture["sanitize_evidence"](
        oversized,
        _json_max_size=32,
    ) == "[REDACTED]"


def test_capture_sanitizer_rejects_encoded_keys_delimiters_and_export_assignments() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    redacted = capture["REDACTED"]

    encoded_mapping = sanitize({
        "%61pi_key": "mapping-secret",
        "%2561uthorization": "header-secret",
        "headers": {"%2541uthorization": "nested-header-secret"},
        "x_api_key_backup": "wrapped-name-secret",
        "%70ath": "/v1/models",
    })
    assert encoded_mapping["%61pi_key"] == redacted
    assert encoded_mapping["%2561uthorization"] == redacted
    assert encoded_mapping["headers"]["%2541uthorization"] == redacted
    assert encoded_mapping["x_api_key_backup"] == redacted
    assert encoded_mapping["%70ath"] == "/v1/models"

    dangerous_scalars = (
        "https://user%3Auserinfo-secret%40example.invalid/v1",
        "https://example.invalid/v1%3Fapi_key%3Dquery-secret",
        "https://example.invalid/v1?api_key%253Dencoded-query-secret",
        "export PGPASSWORD=export-secret",
        "export PGPASSWORD whitespace-secret",
        "env PGPASSWORD=env-secret command",
        "declare -x PGPASSWORD=declare-secret",
        "readonly PGPASSWORD=readonly-secret",
    )
    for value in dangerous_scalars:
        result = sanitize(value)
        assert "secret" not in result
        assert redacted in result

    too_deep = "https://example.invalid/v1?api_key=deep-secret"
    for _ in range(capture["JSON_SANITIZE_MAX_DEPTH"] + 2):
        too_deep = urllib.parse.quote(too_deep, safe="")
    assert sanitize(too_deep) == redacted

    safe_encoded_url = "https://example.invalid/a%20safe%20path?mode=readonly"
    assert sanitize(safe_encoded_url) == safe_encoded_url

    controls = (EVIDENCE / "test_negative_controls.py").read_text()
    assert "def sanitizer_negative_controls()" in controls
    for required_case in (
        "%2561uthorization",
        "userinfo-secret%40",
        "%3Fapi_key%3D",
        "export PGPASSWORD=",
        'capture["JSON_SANITIZE_MAX_DEPTH"] + 2',
    ):
        assert required_case in controls


def test_capture_sanitizer_preserves_canonical_live_process_receipts() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    container = "sparkrun_3d13e8eba3fa512a_4a934bf86599_node_0"
    command = capture["expected_runtime_command"](0)
    wrapper = (
        "bash /workspace/mods/glm-5.3-flash-exl3-upstream-850k/serve_wrapper.sh "
        + command.removeprefix("/usr/bin/python3 /usr/local/bin/vllm serve ")
    )
    inventory = (
        capture["CONTAINER_LAUNCHER_COMMAND"],
        capture["CONTAINER_SHELL_COMMAND"],
        "sleep infinity",
        wrapper,
        command,
        capture["WATCHDOG_COMMAND"],
        "/usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(68)",
        "VLLM::EngineCore",
        "VLLM::Worker_TP0",
        "sleep 5",
    )
    direct = capture["direct_listener_command"](
        container, "192.168.178.47", 0
    )
    canonical = {
        "docker_top": {
            "stdout": (
                "PID PPID PGID SID COMMAND\n"
                + "".join(
                    f"{1234 + index} 1 {1234 + index} {1234 + index} {item}\n"
                    for index, item in enumerate(inventory)
                )
            )
        },
        "serving_process": {"stdout": f"pid=1235\ncommand={command}\n"},
        "direct_listener": {
            "argv": [
                "ssh", "-o", "BatchMode=yes", "192.168.178.47",
                "bash", "-lc", shlex.quote(direct),
            ],
            "stdout": json.dumps({
                "bind": "0.0.0.0", "port": 8000, "inode": "12345",
                "pid": 1235, "command": command,
            }, indent=2),
        },
        "runtime_env": {
            "argv": [
                "ssh", "-o", "BatchMode=yes", "192.168.178.47",
                "bash", "-lc", shlex.quote(
                    f"docker exec {container} bash -lc "
                    f"{shlex.quote(capture['runtime_env_command']())}"
                ),
            ]
        },
        "runtime_overlay": {
            "argv": [
                "ssh", "-o", "BatchMode=yes", "192.168.178.47",
                "bash", "-lc",
                shlex.quote(capture["runtime_overlay_command"](container)),
            ]
        },
    }
    assert capture["sanitize_evidence"](canonical) == canonical
    marker = "synthetic-credential-marker"
    poisoned_stdout = json.loads(json.dumps(canonical))
    poisoned_stdout["docker_top"]["stdout"] += (
        f"9999                helper --api-key {marker}\n"
    )
    poisoned_argv = json.loads(json.dumps(canonical))
    poisoned_argv["direct_listener"]["argv"][-1] += f" --api-key {marker}"
    assert marker not in json.dumps(capture["sanitize_evidence"](poisoned_stdout))
    assert marker not in json.dumps(capture["sanitize_evidence"](poisoned_argv))


def test_capture_sanitizer_closes_composed_and_structural_bypasses() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "reviewer-composed-sensitive-marker"

    mixed = (
        f"PGPASSWORD={marker}\n"
        f"https://example.invalid/v1%3Fapi_key%3D{marker}\n"
        + urllib.parse.quote(json.dumps(json.dumps({"X-Api-Key": marker})), safe="")
    )
    mixed_inline = (
        f"prefix Authorization: Bearer *** suffix "
        + urllib.parse.quote(
            urllib.parse.quote(json.dumps({"api_key": marker}), safe=""),
            safe="",
        )
    )
    header_sequences = {
        "headers": [
            ["Authorization", marker],
            {"name": "Authorization", "value": marker},
        ]
    }
    encoded_argv = {"probe_argv": ["tool", "--api%2Dkey", marker]}
    standard_env = {
        "MYSQL_PWD": marker,
        "DATABASE_URL": marker,
        "DB_URL": marker,
        "POSTGRES_URL": marker,
        "POSTGRESQL_URL": marker,
        "REDIS_URL": marker,
        "MONGODB_URI": marker,
        "MONGO_URI": marker,
        "PGURI": marker,
        "JDBC_URL": marker,
    }

    for case in (mixed, mixed_inline, header_sequences, encoded_argv, standard_env):
        assert marker not in json.dumps(sanitize(case))

    deeply_serialized = "[" * 1500 + json.dumps(marker) + "]" * 1500
    assert sanitize(deeply_serialized) == capture["REDACTED"]

    nested: object = marker
    for _ in range(1500):
        nested = [nested]
    sanitized_nested = sanitize(nested)
    cursor = sanitized_nested
    for _ in range(capture["STRUCTURE_SANITIZE_MAX_DEPTH"]):
        assert isinstance(cursor, list) and len(cursor) == 1
        cursor = cursor[0]
    assert cursor == capture["REDACTED"]


def test_capture_sanitizer_closes_extended_context_credential_bypasses() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "extended-review-sensitive-marker"

    mapping_names = (
        "backupapikeycopy",
        "AZURE_STORAGE_ACCOUNT_KEY",
        "DOCKER_AUTH_CONFIG",
        "SQLALCHEMY_DATABASE_URI",
        "CELERY_BROKER_URL",
        "AMQP_URL",
        "RABBITMQ_URL",
        "KAFKA_URL",
        "ELASTICSEARCH_URL",
        "OPENSEARCH_URL",
        "MSSQL_URL",
        "MYSQL_URL",
        "EXAMPLE_SERVICE_URL",
        "EXAMPLE_SERVICE_URI",
        "EXAMPLE_SERVICE_DSN",
        "backupsecretcopy",
        "backuptokencopy",
        "service_url",
    )
    mapping_cases = [{name: marker} for name in mapping_names]
    mapping_cases += [
        {urllib.parse.quote(urllib.parse.quote(name, safe=""), safe=""): marker}
        for name in mapping_names
    ]

    scalar_cases = (
        f"https://example.invalid/?backupapikeycopy={marker}",
        f"setenv API_KEY {marker}",
        f"printf -v API_KEY %s {marker}",
        f"prefix Authorization: Bearer {marker} suffix",
        f"prefix Proxy-Authorization: Basic {marker} suffix",
        f"curl -u {marker} https://example.invalid/",
        f"curl --proxy-user {marker} https://example.invalid/",
        f"curl -u{marker} https://example.invalid/",
    )

    argv_options = (
        "--database-url",
        "--db-url",
        "--mysql-pwd",
        "--proxy-authorization",
        "--azure-storage-account-key",
        "--docker-auth-config",
        "--user",
        "--proxy-user",
        "--oauth2-bearer",
        "-u",
        "--example-service-url",
        "--example-service-uri",
        "--example-service-dsn",
    )
    argv_cases = [
        {"probe_argv": ["tool", option, marker]}
        for option in argv_options
    ] + [
        {
            "probe_argv": [
                "tool",
                urllib.parse.quote(option, safe="").replace("-", "%2D", 1),
                marker,
            ]
        }
        for option in argv_options
    ]
    argv_cases.append({"probe_argv": ["curl", "-u" + marker]})

    header_cases = (
        {"headers": {"name": "Authorization", "value": marker}},
        {"http_headers": {"name": "Authorization", "value": marker}},
        {"headers": [{"name": "Authorization", "data": marker}]},
        {"headers": ["Authorization", marker]},
        {"header": ["Authorization", marker]},
        {"headers": [f"prefix Authorization: Bearer {marker} suffix"]},
    )

    cases = (*mapping_cases, *scalar_cases, *argv_cases, *header_cases)
    assert len(cases) == 77
    for index, case in enumerate(cases):
        sanitized = sanitize(case)
        assert marker not in json.dumps(sanitized), index
        assert sanitize(sanitized) == sanitized, index

    safe = {
        "path": "/v1/models",
        "requested_url": "http://127.0.0.1:8000/v1/models",
        "max_num_batched_tokens": 7168,
        "headers": {"name": "Content-Type", "value": "application/json"},
    }
    assert sanitize(safe) == safe


def test_capture_sanitizer_redacts_contextual_cli_credential_names() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "contextual-cli-private-marker"
    cases = (
        {"oauth2_bearer": marker},
        {"proxy_user": marker},
        f"https://example.invalid/?oauth2_bearer={marker}",
        f"https://example.invalid/?proxy%5Fuser={marker}",
        json.dumps({"oauth2_bearer": marker}),
        urllib.parse.quote(json.dumps({"proxy_user": marker}), safe=""),
    )
    for index, case in enumerate(cases):
        sanitized = sanitize(case)
        assert marker not in json.dumps(sanitized), index
        assert sanitize(sanitized) == sanitized, index


def test_capture_sanitizer_fails_closed_on_malformed_percent_escapes() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "malformed-percent-private-marker"
    cases = (
        {"api%ZZ_key": marker},
        {"secr%ZZet": marker},
        f"https://example.invalid/?api%ZZ_key={marker}",
        f"https://example.invalid/?secr%25ZZet={marker}",
    )
    for index, case in enumerate(cases):
        sanitized = sanitize(case)
        assert marker not in json.dumps(sanitized), index
        assert sanitize(sanitized) == sanitized, index


def test_capture_sanitizer_redacts_structured_separate_argv_values() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "structured-argv-private-marker"
    cases = (
        {"probe_argv": ["tool", "--api-key", {"payload": marker}]},
        {"probe_argv": ["tool", "--proxy-user", ["payload", marker]]},
        {"probe_argv": ["tool", "%2D%2Doauth2-bearer", (marker,)]},
    )
    for index, case in enumerate(cases):
        sanitized = sanitize(case)
        assert sanitized["probe_argv"][2] == capture["REDACTED"], index
        assert marker not in json.dumps(sanitized), index
        assert sanitize(sanitized) == sanitized, index


def test_capture_sanitizer_redacts_complete_multitoken_header_values() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "multitoken-header-private-marker"
    cases = (
        f"X-Api-Key: synthetic-prefix {marker}",
        f"prefix X-Auth-Token: synthetic-scheme {marker} trailing-text",
        f"Cookie: synthetic-name=synthetic-value; private={marker}",
        f'prefix Authorization: Digest username="synthetic", nonce="{marker}"',
        f'prefix Proxy-Authorization: Digest realm="synthetic", response="{marker}"',
    )
    for index, case in enumerate(cases):
        sanitized = sanitize(case)
        assert marker not in sanitized, index
        assert capture["REDACTED"] in sanitized, index
        assert sanitize(sanitized) == sanitized, index
    safe = "prefix Content-Type: application/json; charset=utf-8"
    assert sanitize(safe) == safe


def test_capture_sanitizer_redacts_rfc_tchar_credential_headers() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "rfc-tchar-header-private-marker"
    separators = "!#$%&'*+-.^_`|~"

    cases = []
    for separator in separators:
        field = separator.join(("X", "Api", "Key"))
        scalar = f"{field}: synthetic-prefix {marker} trailing-text"
        redacted_line = f"{field}: {capture['REDACTED']}"
        cases.extend((
            (scalar, redacted_line),
            (f"embedded-prefix {scalar}", f"embedded-prefix {redacted_line}"),
            (
                json.dumps({"line": scalar}),
                json.dumps({"line": redacted_line}, indent=2),
            ),
            (
                {"headers": [scalar, {"name": field, "value": marker}]},
                {
                    "headers": [
                        redacted_line,
                        {"name": field, "value": capture["REDACTED"]},
                    ]
                },
            ),
        ))

    assert len(cases) == 60
    for index, (case, expected) in enumerate(cases):
        sanitized = sanitize(case)
        assert sanitized == expected, index
        assert marker not in json.dumps(sanitized), index
        assert sanitize(sanitized) == sanitized, index


def test_capture_sanitizer_closes_structured_header_composition_gaps() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    redacted = capture["REDACTED"]
    marker = "structured-header-private-marker"

    encoded_containers = ("%68eaders", "%2568eaders", "http%5Fheaders")
    for field in encoded_containers:
        payload = {field: [["Authorization", marker]]}
        expected = {field: [["Authorization", redacted]]}
        assert sanitize(payload) == expected
        assert sanitize(expected) == expected
        serialized = json.dumps(payload)
        assert json.loads(sanitize(serialized)) == expected

    discriminator_keys = (
        "%6Eame", "%256Eame", "n%61me", "%6Bey", "%256Bey", "k%65y",
        "%68eader", "%2568eader", "h%65ader", "%68eaderName",
        "%2568eaderName", "header%4Eame", "header%256Eame",
    )
    for discriminator in discriminator_keys:
        payload = {"headers": [{discriminator: "Authorization", "value": marker}]}
        expected = {
            "headers": [{discriminator: "Authorization", "value": redacted}]
        }
        assert sanitize(payload) == expected
        assert sanitize(expected) == expected

    ambiguous = (
        {"name": "Content-Type", "key": "Authorization", "value": marker},
        {"key": "Content-Type", "header": "Authorization", "raw": marker},
        {
            "header": "Content-Type",
            "headerName": "Authorization",
            "headerValue": marker,
        },
    )
    for item in ambiguous:
        sanitized = sanitize({"headers": [item]})
        assert sanitized == {"headers": [redacted]}
        assert sanitize(sanitized) == sanitized

    extended = (
        (
            {"headers": [{"headerName": "Authorization", "headerValue": marker}]},
            {
                "headers": [
                    {"headerName": "Authorization", "headerValue": redacted}
                ]
            },
        ),
        (
            {"headers": [["Authorization", redacted, marker]]},
            {"headers": [["Authorization", redacted, redacted]]},
        ),
        (
            {"headers": [{"name": "Authorization", "value": redacted, "raw": marker}]},
            {
                "headers": [
                    {"name": "Authorization", "value": redacted, "raw": redacted}
                ]
            },
        ),
        (
            {"headers": [{"name": "Authorization", "raw_value": marker}]},
            {"headers": [{"name": "Authorization", "raw_value": redacted}]},
        ),
    )
    for payload, expected in extended:
        assert sanitize(payload) == expected
        assert sanitize(expected) == expected

    header_aliases = ("headerList", "header_list", "header%4Cist")
    for field in header_aliases:
        payload = {field: [{"headerName": "Authorization", "headerValue": marker}]}
        expected = {
            field: [{"headerName": "Authorization", "headerValue": redacted}]
        }
        assert sanitize(payload) == expected
        assert json.loads(sanitize(json.dumps(payload))) == expected
        assert sanitize(expected) == expected

    argv_aliases = ("argvList", "argv_list", "argv%4Cist")
    for field in argv_aliases:
        payload = {field: ["tool", "--api-key", marker]}
        expected = {field: ["tool", "--api-key", redacted]}
        assert sanitize(payload) == expected
        assert json.loads(sanitize(json.dumps(payload))) == expected
        assert sanitize(expected) == expected


def test_capture_sanitizer_redacts_nonconvergent_argv_records() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    redacted = capture["REDACTED"]
    marker = "nonconvergent-argv-sensitive-marker"

    deep_option = "--api%2Dkey"
    for _ in range(capture["JSON_SANITIZE_MAX_DEPTH"] + 1):
        deep_option = deep_option.replace("%", "%25")
    assert not capture["percent_decode_fixed_point"](deep_option)[1]

    cases = (
        {"argv": ["--api%ZZkey", marker]},
        {"argvList": [deep_option, marker]},
        {"argv_list": [deep_option, {"opaque": marker}]},
        json.dumps({"argv": ["--api%ZZkey", marker]}),
        json.dumps({"argvList": [deep_option, marker]}),
    )
    for payload in cases:
        sanitized = sanitize(payload)
        assert marker not in json.dumps(sanitized)
        assert redacted in json.dumps(sanitized)
        assert sanitize(sanitized) == sanitized


def test_capture_sanitizer_redacts_attached_curl_header_arguments() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "attached-curl-header-sensitive-marker"
    attached = (
        f"-HAuthorization:Bearer-{marker}",
        f"-H Authorization:Bearer-{marker}",
        f"-H=Authorization:Bearer-{marker}",
        urllib.parse.quote_plus(f"-HAuthorization:Bearer-{marker}"),
    )
    cases: list = [
        {field: [argument]}
        for field in ("argv", "argvList", "argv_list")
        for argument in attached
    ]
    cases.extend(json.dumps(case) for case in list(cases))
    for payload in cases:
        sanitized = sanitize(payload)
        assert marker not in json.dumps(sanitized)
        assert sanitize(sanitized) == sanitized


def test_capture_sanitizer_propagates_self_describing_structural_contexts() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    redacted = capture["REDACTED"]
    marker = "self-describing-structure-sensitive-marker"

    cases = (
        {"headerName": "Authorization", "headerValue": marker},
        {"field_name": "Authorization", "field_value": marker},
        {"httpHeaderName": "Authorization", "httpHeaderValue": marker},
        {"name": "Authorization", "value": marker},
        {"headers": {"items": [["Authorization", marker]]}},
        {"headers": {"items": [["Authorization", marker]], "raw": marker}},
        {"argv": {"items": ["--api-key", marker]}},
        {"argv": {"items": ["--api-key", marker], "raw": marker}},
        {"headers": {"Authorization": marker, "raw": marker}},
        ["Authorization", marker],
    )
    for payload in (*cases, *(json.dumps(case) for case in cases)):
        sanitized = sanitize(payload)
        assert marker not in json.dumps(sanitized)
        assert redacted in json.dumps(sanitized)
        assert sanitize(sanitized) == sanitized

    benign = (
        {"headerName": "Content-Type", "headerValue": "application/json"},
        {"name": "Content-Type", "value": "application/json"},
        {"headers": {"items": [["Content-Type", "application/json"]]}},
    )
    for payload in benign:
        assert sanitize(payload) == payload


def test_capture_sanitizer_propagates_argv_context_through_nested_structures() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "nested-argv-private-marker"
    deep_option = "--api%2Dkey"
    for _ in range(capture["JSON_SANITIZE_MAX_DEPTH"] + 1):
        deep_option = deep_option.replace("%", "%25")
    assert not capture["percent_decode_fixed_point"](deep_option)[1]

    fields = ("argv", "argvList", "argv_list", "argv%4Cist", "argv%5Flist")
    options = ("--api-key", "--api%ZZkey", deep_option)
    wrappers = (
        lambda option: [["tool", option, marker]],
        lambda option: {"items": ["tool", option, marker]},
        lambda option: {"items": {"items": ["tool", option, marker]}},
        lambda option: [{"items": ["tool", option, marker]}],
        lambda option: ({"items": ["tool", option, marker]},),
    )
    cases = []
    for field in fields:
        for option in options:
            for wrap in wrappers:
                payload = {field: wrap(option)}
                cases.extend((payload, json.dumps(payload)))

    assert len(cases) == 150
    for payload in cases:
        sanitized = sanitize(payload)
        assert marker not in json.dumps(sanitized)
        assert sanitize(sanitized) == sanitized


def test_capture_sanitizer_propagates_argv_context_into_inner_serializations() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "inner-serialized-argv-private-marker"
    deep_option = "--api%2Dkey"
    for _ in range(capture["JSON_SANITIZE_MAX_DEPTH"] + 1):
        deep_option = deep_option.replace("%", "%25")
    assert not capture["percent_decode_fixed_point"](deep_option)[1]

    fields = ("argv", "argvList", "argv_list", "argv%4Cist", "argv%5Flist")
    options = ("--api-key", "--api%ZZkey", deep_option)
    wrappers = (
        lambda item: [item],
        lambda item: [[item]],
        lambda item: {"items": [item]},
        lambda item: {"items": {"items": [item]}},
        lambda item: [{"items": [item]}],
        lambda item: ({"items": [item]},),
    )
    cases = []
    for field in fields:
        for option in options:
            inner = json.dumps(["tool", option, marker])
            serialized = (
                inner,
                urllib.parse.quote(inner, safe=""),
                urllib.parse.quote(urllib.parse.quote(inner, safe=""), safe=""),
            )
            for wrap in wrappers:
                for item in serialized:
                    payload = {field: wrap(item)}
                    cases.extend((payload, json.dumps(payload)))

    assert len(cases) == 540
    for payload in cases:
        sanitized = sanitize(payload)
        assert marker not in json.dumps(sanitized)
        assert sanitize(sanitized) == sanitized


def test_capture_sanitizer_tracks_individually_serialized_argv_options() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "serialized-individual-argv-option-private-marker"
    deep_option = "--api%2Dkey"
    for _ in range(capture["JSON_SANITIZE_MAX_DEPTH"] + 1):
        deep_option = deep_option.replace("%", "%25")
    assert not capture["percent_decode_fixed_point"](deep_option)[1]

    fields = ("argv", "argvList", "argv_list", "argv%4Cist", "argv%5Flist")
    wrappers = (
        lambda item: ["tool", item, marker],
        lambda item: ("tool", item, marker),
        lambda item: {"items": ["tool", item, marker]},
        lambda item: {"items": {"items": ["tool", item, marker]}},
        lambda item: [{"items": ["tool", item, marker]}],
        lambda item: ({"items": ["tool", item, marker]},),
    )
    cases = []
    for field in fields:
        for option in ("--api-key", "--api%ZZkey", deep_option):
            inner = json.dumps(option)
            representations = (
                inner,
                urllib.parse.quote(inner, safe=""),
                urllib.parse.quote(urllib.parse.quote(inner, safe=""), safe=""),
            )
            for wrap in wrappers:
                for representation in representations:
                    payload = {field: wrap(representation)}
                    cases.extend((payload, json.dumps(payload)))

    assert len(cases) == 540
    for payload in cases:
        sanitized = sanitize(payload)
        assert marker not in json.dumps(sanitized)
        assert sanitize(sanitized) == sanitized


def test_capture_sanitizer_tracks_serialized_singleton_argv_options() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "serialized-singleton-argv-option-private-marker"
    fields = ("argv", "argvList", "argv_list", "argv%4Cist", "argv%5Flist")
    cases = []
    inner = json.dumps(["--api-key"])
    representations = (
        inner,
        urllib.parse.quote(inner, safe=""),
        urllib.parse.quote(urllib.parse.quote(inner, safe=""), safe=""),
    )
    for field in fields:
        for representation in representations:
            payload = {field: ["tool", representation, marker]}
            cases.extend((payload, json.dumps(payload)))

    assert len(cases) == 30
    for payload in cases:
        sanitized = sanitize(payload)
        assert marker not in json.dumps(sanitized)
        assert sanitize(sanitized) == sanitized


def test_capture_sanitizer_tracks_nested_serialized_argv_fragments() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "nested-serialized-argv-fragment-private-marker"
    fields = ("argv", "argvList", "argv_list", "argv%4Cist", "argv%5Flist")
    fragments = (
        [["--api-key"]],
        {"items": ["--api-key"]},
        {"items": {"items": ["--api-key"]}},
        [{"items": ["--api-key"]}],
    )
    cases = []
    for field in fields:
        for fragment in fragments:
            inner = json.dumps(fragment)
            representations = (
                inner,
                urllib.parse.quote(inner, safe=""),
                urllib.parse.quote(urllib.parse.quote(inner, safe=""), safe=""),
            )
            for representation in representations:
                payload = {field: ["tool", representation, marker]}
                cases.extend((payload, json.dumps(payload)))

    assert len(cases) == 120
    for payload in cases:
        sanitized = sanitize(payload)
        assert marker not in json.dumps(sanitized)
        assert sanitize(sanitized) == sanitized


def test_capture_sanitizer_tracks_nested_structural_argv_fragments() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "nested-structural-argv-fragment-private-marker"
    fields = ("argv", "argvList", "argv_list", "argv%4Cist", "argv%5Flist")
    fragments = (
        [["--api-key"]],
        {"items": ["--api-key"]},
        {"items": {"items": ["--api-key"]}},
        [{"items": ["--api-key"]}],
    )
    cases = []
    for field in fields:
        for fragment in fragments:
            payload = {field: ["tool", fragment, marker]}
            cases.extend((payload, json.dumps(payload)))

    assert len(cases) == 40
    for payload in cases:
        sanitized = sanitize(payload)
        assert marker not in json.dumps(sanitized)
        assert sanitize(sanitized) == sanitized


def test_capture_sanitizer_taints_sensitive_mapping_sibling_payloads() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    marker = "sensitive-mapping-sibling-private-marker"
    cases = []
    for sibling in ("raw", "value", "raw_value", "data"):
        payload = {"api_key": marker, sibling: marker}
        cases.extend((payload, json.dumps(payload)))

    assert len(cases) == 8
    for payload in cases:
        sanitized = sanitize(payload)
        assert marker not in json.dumps(sanitized)
        assert sanitize(sanitized) == sanitized


def test_capture_sanitizer_rejects_noncanonical_proxy_commands() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    runtime = json.loads((EVIDENCE / "runtime.json").read_text())
    proxy_argv = next(
        row["argv"]
        for row in runtime["proxy_processes"]
        if any(item.endswith("/bin/litellm") for item in row["argv"])
    )
    canonical = " ".join(proxy_argv)
    assert capture["safe_command_value"](canonical)
    assert sanitize(canonical, "command") == canonical

    for index, item in enumerate(proxy_argv[:2]):
        archive_id = item.split("/archive-v0/", 1)[1].split("/", 1)[0]
        mutated_argv = list(proxy_argv)
        mutated_argv[index] = item.replace(
            archive_id, "synthetic-proxy-command-canary", 1
        )
        mutated = " ".join(mutated_argv)
        assert mutated != canonical
        assert sanitize(mutated, "command") == capture["REDACTED"]


def test_detect_secrets_verifier_rejects_duplicate_findings_and_any_stderr(
    tmp_path: Path,
) -> None:
    verifier = runpy.run_path(str(ROOT / "scripts/verify_detect_secrets.py"))
    row = {
        "filename": "fixture.txt",
        "type": "Secret Keyword",
        "hashed_secret": "synthetic-digest",
    }
    document = {"results": {"fixture.txt": [row, dict(row)]}}
    identity = ("fixture.txt", "Secret Keyword", "synthetic-digest")
    assert verifier["identities"](document) == Counter({identity: 2})
    assert verifier["scan_process_errors"](0, "", "synthetic warning") == [
        "detect-secrets wrote stderr"
    ]

    reviewed = json.loads((ROOT / ".secrets.baseline").read_text())
    filename = next(iter(reviewed["results"]))
    reduced = dict(reviewed)
    reduced["results"] = {filename: reviewed["results"][filename]}
    original = verifier["raw_occurrence_summary"](ROOT, reduced)
    copied = tmp_path / filename
    copied.parent.mkdir(parents=True)
    source_lines = (ROOT / filename).read_text().splitlines()
    finding_line = reviewed["results"][filename][0]["line_number"]
    copied.write_text("\n".join((*source_lines, source_lines[finding_line - 1])) + "\n")
    duplicated = verifier["raw_occurrence_summary"](tmp_path, reduced)
    assert duplicated["occurrences"] > original["occurrences"]
    assert duplicated["sha256"] != original["sha256"]


def test_detect_secrets_baseline_is_reviewed_and_reproducible() -> None:
    verifier = ROOT / "scripts/verify_detect_secrets.py"
    baseline = ROOT / ".secrets.baseline"
    assert verifier.is_file()
    assert baseline.is_file()
    result = subprocess.run(
        [sys.executable, str(verifier), "--root", str(ROOT)],
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["findings"] == report["adjudicated_false_positives"]
    assert report["findings"] > 0
    assert report["stderr"] == ""


def test_capture_sanitizer_preserves_reviewed_percent_literal_runtime_argv() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitize = capture["sanitize_evidence"]
    runtime = json.loads((EVIDENCE / "runtime.json").read_text())
    for host in runtime["hosts"].values():
        for key in ("runtime_mod_manifest", "serving_process", "nccl_loaded", "rdma"):
            argv = host[key]["argv"]
            assert sanitize({"probe_argv": argv})["probe_argv"] == argv, key


def test_runtime_evidence_is_sanitizer_idempotent() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    runtime = json.loads((EVIDENCE / "runtime.json").read_text())
    assert capture["sanitize_evidence"](runtime) == runtime


def test_listener_probes_use_real_namespaces_without_sudo() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    assert verifier["LISTENER_PROBE"] == capture["LISTENER_PROBE"]
    assert verifier["DIRECT_LISTENER_PROBE"] == capture["DIRECT_LISTENER_PROBE"]
    host0, host1 = "192.168.178.47", "192.168.178.46"
    container0, container1 = "sparkrun_example_node_0", "sparkrun_example_node_1"
    for module in (capture, verifier):
        head = module["direct_listener_command"](container0, host0, 0)
        worker = module["direct_listener_command"](container1, host1, 1)
        assert head.startswith(f"docker exec {container0} /usr/bin/python3 -S -c ")
        assert worker.startswith("/usr/bin/python3 -S -c ")
        assert "docker exec" not in worker
        assert "sudo" not in head + worker


def test_pid_namespace_receipt_joins_host_and_container_pids() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    host = "192.168.178.47"
    host_pid, container_pid = 2932817, 676
    command = verifier["pid_namespace_command"](host_pid)
    receipt = {
        "argv": verifier["expected_remote_argv"](host, command),
        "returncode": 0,
        "stdout": f"host_pid={host_pid}\ncontainer_pid={container_pid}\n",
        "stderr": "",
    }
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    sanitized = capture["sanitize_evidence"]({"pid_namespace": receipt})
    assert sanitized["pid_namespace"] == receipt
    assert verifier["pid_namespace_failures"](
        receipt, host, host_pid, container_pid
    ) == []
    for bad_host, bad_container in ((host_pid + 1, container_pid), (host_pid, container_pid + 1)):
        assert verifier["pid_namespace_failures"](
            receipt, host, bad_host, bad_container
        )


def test_serve_marker_receipt_is_structured_and_fatal_closed() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    assert verifier["SERVE_MARKER_PROBE"] == capture["SERVE_MARKER_PROBE"]
    host = "192.168.178.47"
    container = "sparkrun_example_node_0"
    payload = {
        "required_marker_counts": {
            marker: 1 for marker in verifier["REQUIRED_SERVE_MARKERS"]
        },
        "e3_markers": [{
            "grouped_calls": 7,
            "fat_expert_runs": 11,
            "configured_tier": "grouped",
            "effective_tier": "grouped",
        }],
        "fatal_matches": [],
    }
    command = verifier["serve_marker_command"](container)
    receipt = {
        "argv": verifier["expected_remote_argv"](host, command),
        "returncode": 0,
        "stdout": json.dumps(payload, sort_keys=True) + "\n",
        "stderr": "",
    }
    assert verifier["serve_marker_failures"](receipt, host, container) == []
    missing = json.loads(json.dumps(payload))
    missing["required_marker_counts"][verifier["REQUIRED_SERVE_MARKERS"][0]] = 0
    fatal = json.loads(json.dumps(payload))
    fatal["fatal_matches"] = ["CUDA error: an illegal memory access"]
    wrong_tier = json.loads(json.dumps(payload))
    wrong_tier["e3_markers"][0]["effective_tier"] = "fallback"
    for changed in (missing, fatal, wrong_tier):
        mutated = dict(receipt, stdout=json.dumps(changed, sort_keys=True) + "\n")
        assert verifier["serve_marker_failures"](mutated, host, container)


def test_runtime_capture_and_verifier_require_wrapper_lineage_receipts() -> None:
    capture = (EVIDENCE / "capture_runtime.py").read_text()
    verifier = (EVIDENCE / "verify.py").read_text()
    assert "wrapper_lineage = remote(host, wrapper_lineage_command(wrapper_host_pid))" in capture
    assert '"wrapper_lineage": wrapper_lineage' in capture
    assert "serve_markers = remote(host, serve_marker_command(container))" in capture
    assert '"serve_markers": serve_markers' in capture
    assert '"serving_process", "pid_namespace", "wrapper_lineage"' in verifier
    assert '"serve_log", "serve_markers", "runtime_env"' in verifier
    assert "wrapper_lineage_failures(" in verifier
    controls = (EVIDENCE / "test_negative_controls.py").read_text()
    assert 'row["wrapper_lineage"]' in controls
    assert 'row["serve_markers"]' in controls


def test_wrapper_lineage_binds_external_shim_and_pid_namespace() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    assert verifier["WRAPPER_LINEAGE_PROBE"] == capture["WRAPPER_LINEAGE_PROBE"]
    host = "192.168.178.47"
    wrapper_host_pid = 2000
    parent_host_pid = 9000
    container_id = "a" * 64
    command = verifier["wrapper_lineage_command"](wrapper_host_pid)
    payload = {
        "wrapper_host_pid": wrapper_host_pid,

        "parent_host_pid": parent_host_pid,
        "parent_executable": "/usr/bin/containerd-shim-runc-v2",
        "parent_namespace": "moby",
        "parent_container_id": container_id,
        "parent_address": "/run/containerd/containerd.sock",
    }
    receipt = {
        "argv": verifier["expected_remote_argv"](host, command),
        "returncode": 0,
        "stdout": json.dumps(payload, sort_keys=True) + "\n",
        "stderr": "",
    }
    assert verifier["wrapper_lineage_failures"](
        receipt, host, wrapper_host_pid, parent_host_pid, container_id
    ) == []
    sanitized = capture["sanitize_evidence"]({"wrapper_lineage": receipt})[
        "wrapper_lineage"
    ]
    assert sanitized["argv"] == receipt["argv"]
    assert json.loads(sanitized["stdout"]) == payload
    assert sanitized["returncode"] == 0 and sanitized["stderr"] == ""
    for field, value in (
        ("wrapper_host_pid", 2999),
        ("parent_host_pid", 9001),
        ("parent_executable", "/tmp/shim"),
        ("parent_namespace", "other"),
        ("parent_container_id", "b" * 64),
        ("parent_address", "/tmp/containerd.sock"),
    ):
        changed = dict(payload, **{field: value})
        mutated = dict(receipt, stdout=json.dumps(changed, sort_keys=True) + "\n")
        assert verifier["wrapper_lineage_failures"](
            mutated, host, wrapper_host_pid, parent_host_pid, container_id
        ), field


def test_docker_top_inventory_is_closed_and_rank_bound() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    contract = verifier["reviewed_recipe_command"](RECIPE)

    def canonical(rank: int, include_transient_sleep: bool = True) -> str:
        rows = [
            (1000, 9000, 1000, 1000, verifier["CONTAINER_LAUNCHER_COMMAND"]),
            (1001, 1000, 1000, 1000, verifier["CONTAINER_SHELL_COMMAND"]),
            (1002, 1001, 1000, 1000, "sleep infinity"),
            (2000, 9000, 2000, 2000, verifier["expected_wrapper_command"](rank, contract)),
            (2001, 2000, 2001, 2001, verifier["expected_runtime_command"](rank, contract)),
            (2002, 9000, 2002, 2002, verifier["WATCHDOG_COMMAND"]),
            (2003, 2001, 2001, 2001, "/usr/bin/python3 -c from multiprocessing.resource_tracker import main;main(68)"),
        ]
        if rank == 0:
            rows.extend(((2004, 2001, 2001, 2001, "VLLM::EngineCore"), (2005, 2004, 2001, 2001, "VLLM::Worker_TP0")))
        else:
            rows.append((2005, 2001, 2001, 2001, "VLLM::Worker_TP1"))
        if include_transient_sleep:
            rows.append((2006, 2002, 2002, 2002, "sleep 5"))
        return "PID PPID PGID SID COMMAND\n" + "".join(
            f"{pid} {ppid} {pgid} {sid} {command}\n"
            for pid, ppid, pgid, sid, command in rows
        )

    validate = verifier["docker_top_inventory_failures"]
    for rank in (0, 1):
        listener_pid = 676 if rank == 0 else None
        aligned = canonical(rank).replace(
            "PID PPID PGID SID COMMAND",
            "PID                 PPID                PGID                SID                 COMMAND",
            1,
        )
        assert validate(aligned, rank, 2001, 676, listener_pid, 676, contract) == []
        assert validate(canonical(rank), rank, 2001, 676, listener_pid, 676, contract) == []
        assert validate(canonical(rank, False), rank, 2001, 676, listener_pid, 676, contract) == []

    wrapper_mutation = canonical(0).replace("--port 8000", "--port 8001", 1)
    assert "docker top closed process inventory" in validate(wrapper_mutation, 0, 2001, 676, 676, 676, contract)
    rank_mutation = canonical(1).replace("--node-rank 1", "--node-rank 0")
    assert "docker top closed process inventory" in validate(rank_mutation, 1, 2001, 676, None, 676, contract)
    for name, mutated in {
        "vllm pid": canonical(0).replace("2001 2000 2001 2001 /usr/bin/python3", "2999 2000 2999 2999 /usr/bin/python3", 1),
        "runtime parent": canonical(0).replace("2001 2000 2001 2001 /usr/bin/python3", "2001 1001 2001 2001 /usr/bin/python3", 1),
        "serve pgid": canonical(0).replace("2001 2000 2001 2001 /usr/bin/python3", "2001 2000 2999 2001 /usr/bin/python3", 1),
        "serve session": canonical(0).replace("2001 2000 2001 2001 /usr/bin/python3", "2001 2000 2001 2999 /usr/bin/python3", 1),
        "wrapper group": canonical(0).replace("2000 9000 2000 2000 bash /workspace", "2000 9000 2999 2000 bash /workspace", 1),
        "shared shim": canonical(0).replace("1000 9000 1000 1000 bash -c", "1000 9001 1000 1000 bash -c", 1),
        "launcher group": canonical(0).replace("1000 9000 1000 1000 bash -c", "1000 9000 2999 1000 bash -c", 1),
        "container shell parent": canonical(0).replace("1001 1000 1000 1000 bash --", "1001 9999 1000 1000 bash --", 1),
        "container shell session": canonical(0).replace("1001 1000 1000 1000 bash --", "1001 1000 1000 2999 bash --", 1),
        "keepalive parent": canonical(0).replace("1002 1001 1000 1000 sleep infinity", "1002 9999 1000 1000 sleep infinity", 1),
        "watchdog parent": canonical(0).replace("2002 9000 2002 2002 bash -c SERVE_PID", "2002 9999 2002 2002 bash -c SERVE_PID", 1),
        "watchdog group": canonical(0).replace("2002 9000 2002 2002 bash -c SERVE_PID", "2002 9000 2999 2002 bash -c SERVE_PID", 1),
        "resource tracker parent": canonical(0).replace("2003 2001 2001 2001 /usr/bin/python3 -c from multiprocessing", "2003 9999 2001 2001 /usr/bin/python3 -c from multiprocessing", 1),
        "engine parent": canonical(0).replace("2004 2001 2001 2001 VLLM::EngineCore", "2004 9999 2001 2001 VLLM::EngineCore", 1),
        "worker parent": canonical(0).replace("2005 2004 2001 2001 VLLM::Worker_TP0", "2005 9999 2001 2001 VLLM::Worker_TP0", 1),
        "worker group": canonical(0).replace("2005 2004 2001 2001 VLLM::Worker_TP0", "2005 2004 2999 2001 VLLM::Worker_TP0", 1),
        "watchdog sleep parent": canonical(0).replace("2006 2002 2002 2002 sleep 5", "2006 9999 2002 2002 sleep 5", 1),
    }.items():
        assert "docker top process identity join" in validate(
            mutated, 0, 2001, 676, 676, 676, contract
        ), name
    unknown = canonical(1) + "9999 1 9999 9999 /usr/bin/nc -l -p 9999\n"
    assert "docker top closed process inventory" in validate(unknown, 1, 2001, 676, None, 676, contract)


def test_worker_rank_requires_exact_negative_listener_receipt() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    validate = verifier["worker_listener_failures"]
    host = "192.168.178.46"
    container = "sparkrun_example_node_1"
    command = verifier["direct_listener_command"](container, host, 1)
    receipt = {
        "argv": verifier["expected_remote_argv"](host, command),
        "returncode": 0,
        "stdout": json.dumps({
            "namespace": "host",
            "host": host,
            "container": container,
            "rank": 1,
            "endpoint": {"port": 8000},
            "listeners": [],
        }, sort_keys=True) + "\n",
        "stderr": "",
    }
    assert validate(receipt, host, container) == []
    for mutation in (
        None,
        {},
        dict(receipt, stdout="null\n"),
        dict(receipt, stdout=json.dumps({**json.loads(receipt["stdout"]), "listeners": [{"pid": 7}]}) + "\n"),
        dict(receipt, stdout=json.dumps({**json.loads(receipt["stdout"]), "rank": 0}) + "\n"),
    ):
        assert "worker negative listener receipt" in validate(mutation, host, container)


def test_lifecycle_suite_mandates_sigint_cleanup_on_both_ranks() -> None:
    suite = (MOD / "test_serve_wrapper_supervision.sh").read_text()
    control = MOD / "test_serve_wrapper_sigint.py"
    assert control.is_file()
    assert 'python3 "$MOD_DIR/test_serve_wrapper_sigint.py"' in suite
    result = subprocess.run(
        [sys.executable, str(control)], text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SIGINT cleanup control OK: rank=0 rc=130" in result.stdout
    assert "SIGINT cleanup control OK: rank=1 rc=130" in result.stdout


def test_runtime_mod_manifest_is_an_external_launch_and_per_rank_trust_root() -> None:
    capture = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    expected = verifier["EXPECTED_MOD_MANIFEST_SHA256"]
    assert re.fullmatch(r"[0-9a-f]{64}", expected)
    assert hashlib.sha256((MOD / "SHA256SUMS").read_bytes()).hexdigest() == expected
    assert capture["EXPECTED_MOD_MANIFEST_SHA256"] == expected
    command = capture["runtime_mod_manifest_command"]("example-container", expected)
    assert "sha256sum -c SHA256SUMS" in command
    assert expected in command
    assert 'parser.add_argument("--expected-mod-manifest-sha256", required=True)' in (
        EVIDENCE / "capture_runtime.py"
    ).read_text()
    verifier_source = (EVIDENCE / "verify.py").read_text()
    assert "runtime_mod_manifest" in verifier_source
    assert "launch mod manifest binding" in verifier_source
    controls = (EVIDENCE / "test_negative_controls.py").read_text()
    assert "executable mod substituted with regenerated manifest" in controls


def test_image_only_surface_has_hash_bound_direct_and_proxy_ocr() -> None:
    recipe = load_recipe()
    exact_limit = "--limit-mm-per-prompt '{\"image\":4,\"video\":0}'"
    assert exact_limit in recipe["command"]
    command_contract = runpy.run_path(str(EVIDENCE / "command_contract.py"))
    rank_argv = command_contract["reviewed_rank_argv"](RECIPE)
    for rank in (0, 1):
        argv = rank_argv[rank]
        limit_index = argv.index("--limit-mm-per-prompt")
        assert argv[limit_index + 1] == '{"image":4,"video":0}'
    acceptance = runpy.run_path(str(EVIDENCE / "acceptance.py"))
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    fixture = EVIDENCE / "vision-ocr.png"
    expected = verifier["canonical_ocr_fixture"]()
    assert fixture.read_bytes() == expected
    assert hashlib.sha256(expected).hexdigest() == verifier["OCR_FIXTURE_SHA256"]
    assert acceptance["ocr_request"](expected) == verifier["ocr_request"](expected)
    assert {"direct_ocr", "proxy_ocr"}.issubset(verifier["EXPECTED_CHECKS"])
    controls = (EVIDENCE / "test_negative_controls.py").read_text()
    assert "OCR fixture and both inline requests replaced self-consistently" in controls
    assert "direct OCR canonical request substituted self-consistently" in controls


def test_video_zero_errors_match_pinned_live_envelopes() -> None:
    acceptance = runpy.run_path(str(EVIDENCE / "acceptance.py"))
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    message = (
        "At most 0 video(s) may be provided in one prompt. "
        "Set `--limit-mm-per-prompt` to increase this limit. (parameter=video)"
    )
    direct = {
        "error": {
            "message": message,
            "type": "BadRequestError",
            "param": "video",
            "code": 400,
        }
    }
    proxy = {
        "error": {
            "message": (
                "litellm.BadRequestError: OpenAIException - " + message
                + ". Received Model Group=GLM-5.3-Flash-EXL3\n"
                "Available Model Group Fallbacks=None"
            ),
            "type": None,
            "param": None,
            "code": "400",
        }
    }
    for module in (acceptance, verifier):
        assert module["VIDEO_LIMIT_ERROR"] == direct
        assert module["PROXY_VIDEO_LIMIT_ERROR"] == proxy


def test_acceptance_records_exact_canonical_video_fixture_metadata() -> None:
    source = (EVIDENCE / "acceptance.py").read_text()
    assert '"mime_type": "image/gif"' in source
    assert (
        '"description": "deterministic two-frame 224x224 inline GIF for video=0 rejection"'
        in source
    )


def test_video_zero_surface_has_hash_bound_direct_and_proxy_rejections() -> None:
    acceptance = runpy.run_path(str(EVIDENCE / "acceptance.py"))
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    fixture = EVIDENCE / "video-tiny.gif"
    expected = verifier["canonical_video_fixture"]()
    assert fixture.read_bytes() == expected
    assert len(expected) < 1024
    assert hashlib.sha256(expected).hexdigest() == verifier["VIDEO_FIXTURE_SHA256"]
    body = verifier["video_rejection_request"](expected)
    assert acceptance["video_rejection_request"](expected) == body
    assert body["model"] == "GLM-5.3-Flash-EXL3"
    assert body["messages"][0]["content"][1]["type"] == "video_url"
    assert {"direct_video_rejected", "proxy_video_rejected"}.issubset(
        verifier["EXPECTED_CHECKS"]
    )

    for key in ("direct_video_rejected", "proxy_video_rejected"):
        base = "http://127.0.0.1:4000" if key.startswith("proxy") else "http://127.0.0.1:8000"
        row = {
            "passed": True,
            "requested_url": base + "/v1/chat/completions",
            "effective_url": base + "/v1/chat/completions",
            "path": "/v1/chat/completions",
            "http": 400,
            "request": body,
            "request_sha256": verifier["request_sha"](body),
            "response": (
                verifier["PROXY_VIDEO_LIMIT_ERROR"]
                if key.startswith("proxy")
                else verifier["VIDEO_LIMIT_ERROR"]
            ),
        }
        assert verifier["video_rejection_failures"](key, row, expected) == []
        for field, value in (("passed", False), ("http", 200)):
            changed = dict(row, **{field: value})
            assert verifier["video_rejection_failures"](key, changed, expected)


def test_launch_contract_predeclares_deterministic_container_name() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    argv = verifier["EXPECTED_LAUNCH_ARGV"]
    assert argv[-2:] == ["--container-name", verifier["CLUSTER"]]


def test_verifier_binds_recipe_raw_command_to_status_and_launch_receipt() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    contract = verifier["reviewed_recipe_command"](RECIPE)
    raw = contract["raw_command"]
    receipt = {"recipe_command": raw}
    assert verifier["command_binding_failures"](contract, raw, receipt) == []
    assert verifier["command_binding_failures"](
        contract, raw.replace('"video":0', '"video":1'), receipt
    )
    extra = raw + "\necho injected"
    assert verifier["command_binding_failures"](contract, extra, {"recipe_command": extra})
    assert verifier["command_binding_failures"](
        contract, raw, {"recipe_command": raw + " --extra"}
    )


def test_capture_records_relationship_rich_top_and_both_listener_receipts() -> None:
    source = (EVIDENCE / "capture_runtime.py").read_text()
    assert "docker top {qcontainer} -eo pid,ppid,pgid,sid,args" in source
    assert 'host_record["direct_listener"] = remote(' in source
    assert "host, direct_listener_command(container, host, rank)" in source
    assert 'if host == HOSTS[0]:\n            host_record["direct_listener"]' not in source


def test_successful_chat_receipt_rejects_contradictory_inner_verdict() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    body = verifier["exact_request"]("GLM53_DIRECT_OK")
    row = {
        "passed": False,
        "requested_url": "http://127.0.0.1:8000/v1/chat/completions",
        "effective_url": "http://127.0.0.1:8000/v1/chat/completions",
        "path": "/v1/chat/completions",
        "http": 200,
        "request": body,
        "request_sha256": verifier["request_sha"](body),
        "response": {
            "model": verifier["MODEL"],
            "object": "chat.completion",
            "system_fingerprint": verifier["EXPECTED_SYSTEM_FINGERPRINT"],
            "created": 10,
        },
        "started_at": 10.0,
        "completed_at": 10.5,
    }
    failures: list[str] = []
    verifier["verify_chat_receipt"]("direct_exact", row, body, failures)
    assert "direct_exact verdict" in failures


def test_publication_docs_record_live_recapture_and_recommend_850k() -> None:
    root_readme = (ROOT / "README.md").read_text()
    evidence_readme = (ROOT / "evidence" / "README.md").read_text()
    bundle_readme = (EVIDENCE / "README.md").read_text()
    recipe = RECIPE.read_text()
    assert "Fresh live validation: **PASSED**" in root_readme
    assert "**LIVE_CAPTURE_PASSED**" in evidence_readme
    assert "Evidence state: **LIVE_CAPTURE_PASSED; PUBLICATION_PENDING**" in bundle_readme
    assert "release evidence must include exact direct and proxy rejection receipts" in recipe
    assert "recommended GLM 5.3 Flash EXL3 850K recipe" in root_readme
    assert "recommended GLM 5.3 Flash EXL3 1M recipe" not in root_readme
    assert "Maintainer-approved publication exception" not in root_readme
    assert "independent audit returning `passed=false`" not in root_readme
    assert "remain unresolved" not in root_readme
    assert "post-publication verification remains pending" in evidence_readme.lower()


def test_multitoken_stop_gate_runs_on_host_and_runtime_keeps_production_path() -> None:
    gate = MOD / "test_suppress_stops_multitoken.py"
    result = subprocess.run(
        [sys.executable, str(gate)], text=True, capture_output=True, cwd=MOD
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "synthetic host self-test OK" in result.stdout
    capture = (EVIDENCE / "capture_runtime.py").read_text()
    assert "test_suppress_stops_multitoken.py --production" in capture
    suite = (MOD / "run_upstream_compatibility_suite.sh").read_text()
    assert "test_suppress_stops_multitoken.py" in suite


def test_acceptance_records_invocation_and_binds_every_receipt_route() -> None:
    acceptance_source = (EVIDENCE / "acceptance.py").read_text()
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    assert '"invocation_argv": sys.argv' in acceptance_source
    assert '"effective_url"' in acceptance_source
    assert '"path": path' in acceptance_source
    direct = {
        "requested_url": "http://127.0.0.1:8000/v1/chat/completions",
        "effective_url": "http://127.0.0.1:8000/v1/chat/completions",
        "path": "/v1/chat/completions",
    }
    proxy = {
        "requested_url": "http://127.0.0.1:4000/v1/models",
        "effective_url": "http://127.0.0.1:4000/v1/models",
        "path": "/v1/models",
    }
    assert verifier["receipt_route_failures"]("direct_exact", direct) == []
    assert verifier["receipt_route_failures"]("proxy_models", proxy) == []
    for field in ("requested_url", "effective_url"):
        forged = dict(direct, **{field: "http://127.0.0.1:4000/v1/chat/completions"})
        assert verifier["receipt_route_failures"]("direct_exact", forged) == [
            "direct_exact requested/effective URL/path"
        ]
    assert "requested URL rebound to proxy port with refreshed manifest" in (
        EVIDENCE / "test_negative_controls.py"
    ).read_text()
    assert "EXPECTED_ACCEPTANCE_ARGV" in verifier


def test_acceptance_receipt_intervals_are_complete_ordered_and_contained() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    checks: dict[str, dict] = {
        key: {"started_at": 110.0, "completed_at": 120.0}
        for key in verifier["EXPECTED_CHECKS"]
        if key != "direct_concurrency_c4"
    }
    checks["direct_concurrency_c4"] = {
        "results": [
            {"started_at": 111.0 + index, "completed_at": 121.0 + index}
            for index in range(4)
        ]
    }
    acceptance = {"started_at": 100.0, "completed_at": 200.0, "checks": checks}
    validate = verifier["receipt_interval_failures"]
    assert validate(acceptance, 90) == []

    cases = {
        "pre-launch": {"started_at": 80.0, "completed_at": 95.0},
        "reversed": {"started_at": 130.0, "completed_at": 120.0},
        "missing": {"started_at": 130.0},
        "out-of-window": {"started_at": 190.0, "completed_at": 210.0},
    }
    for name, replacement in cases.items():
        mutated = json.loads(json.dumps(acceptance))
        mutated["checks"]["direct_exact"] = replacement
        failures = validate(mutated, 90)
        assert "direct_exact request receipt interval" in failures, (name, failures)
    missing_row = json.loads(json.dumps(acceptance))
    missing_row["checks"]["direct_concurrency_c4"]["results"].pop()
    assert "request receipt cardinality" in validate(missing_row, 90)
    controls = (EVIDENCE / "test_negative_controls.py").read_text()
    for control in (
        "request receipt starts before launch",
        "request receipt interval reversed",
        "request receipt timestamp missing",
        "request receipt completes outside acceptance window",
        "request receipt cardinality reduced",
    ):
        assert control in controls


def test_launch_epoch_binds_immutable_receipt_and_docker_creation() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    epoch = 1_788_961_176
    contract = verifier["reviewed_recipe_command"](RECIPE)
    receipt = {
        "schema": 1,
        "cluster_id": verifier["CLUSTER"],
        "recipe_sha256": hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
        "recipe_command": contract["raw_command"],
        "mod_manifest_sha256": verifier["EXPECTED_MOD_MANIFEST_SHA256"],
        "launch_epoch": epoch,
        "recorded_at": "2026-09-09T13:39:36+00:00",
        "argv": verifier["EXPECTED_LAUNCH_ARGV"],
    }
    docker_facts = {
        "192.168.178.47": {
            "Created": "2026-09-09T13:39:45.266938426Z",
            "StartedAt": "2026-09-09T13:39:45.318651435Z",
        },
        "192.168.178.46": {
            "Created": "2026-09-09T13:39:45.281265595Z",
            "StartedAt": "2026-09-09T13:39:45.350296072Z",
        },
    }
    assert verifier["launch_epoch_failures"](
        epoch, receipt, docker_facts, receipt["recipe_sha256"], contract
    ) == []
    shifted = dict(
        receipt,
        launch_epoch=1_788_969_764,
        recorded_at="2026-09-09T16:02:44+00:00",
    )
    mismatched_recorded = dict(
        receipt, recorded_at="2026-09-09T13:39:35+00:00"
    )
    assert "immutable launch receipt binding" in verifier["launch_epoch_failures"](
        epoch,
        mismatched_recorded,
        docker_facts,
        receipt["recipe_sha256"],
        contract,
    )
    assert "launch epoch precedes Docker Created" in verifier["launch_epoch_failures"](
        shifted["launch_epoch"], shifted, docker_facts, shifted["recipe_sha256"], contract
    )
    capture = (EVIDENCE / "capture_runtime.py").read_text()
    assert 'parser.add_argument("--launch-receipt", required=True)' in capture
    assert '"launch_receipt_sha256"' in capture


def test_runtime_capture_semantically_binds_nccl_to_roce_and_telemetry() -> None:
    capture_ns = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    verifier_ns = runpy.run_path(str(EVIDENCE / "verify.py"))
    host = "192.168.178.47"
    container = "test-container"
    command = capture_ns["roce_records_command"](container)
    assert command == verifier_ns["roce_records_command"](container)
    assert "verify_roce_gid.py --json" in command
    records = verifier_ns["EXPECTED_ROCE"][host]
    env = "\n".join(
        (
            "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1",
            "NCCL_IB_GID_INDEX=3",
            "NCCL_SOCKET_IFNAME=enP7s7,enp1s0f1np1,enP2p1s0f1np1",
        )
    ) + "\n"
    failures = verifier_ns["roce_binding_failures"](
        host, env, records, {"rocep1s0f1", "roceP2p1s0f1"}
    )
    assert failures == []
    live_json_records = [dict(row) for row in records]
    live_json_records[0]["gid"] = "::ffff:c0a8:348"
    live_json_records[1]["gid"] = "::ffff:c0a8:248"
    assert verifier_ns["roce_binding_failures"](
        host, env, live_json_records, {"rocep1s0f1", "roceP2p1s0f1"}
    ) == []
    forged = [dict(row) for row in records]
    forged[0]["hca"] = "fabricated0"
    assert "runtime RoCE records" in verifier_ns["roce_binding_failures"](
        host, env, forged, {"rocep1s0f1", "roceP2p1s0f1"}
    )


def test_runtime_capture_binds_host_rank_container_pid_and_patch_state() -> None:
    capture = (EVIDENCE / "capture_runtime.py").read_text()
    capture_ns = runpy.run_path(str(EVIDENCE / "capture_runtime.py"))
    verifier_ns = runpy.run_path(str(EVIDENCE / "verify.py"))
    assert "for rank, host in enumerate(HOSTS):" in capture
    assert '"ssh_host": host' in capture
    assert '"rank": rank' in capture
    assert '"serving_process": serving_process' in capture
    assert "verify_runtime_patch_state.py" in capture
    assert "serving_pid" in capture
    assert "/proc/{serving_pid}/maps" in capture
    container = "test-container"
    capture_command = capture_ns["runtime_overlay_command"](container)
    verifier_command = verifier_ns["runtime_overlay_command"](container)
    assert capture_command == verifier_command
    assert "bash -euo pipefail -c" in capture_command
    inner = shlex.split(capture_command)[-1]
    assert "; " not in inner
    assert inner.count(" && ") == 2
    verifier = (EVIDENCE / "verify.py").read_text()
    assert "[OK] complete GLM-5.3 runtime patched state verified" in verifier
    assert "runtime overlay FATAL stderr" in verifier


def test_proxy_capture_binds_status_supervisor_parent_listener_and_upstream_socket() -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    supervisor_pid = 4099
    parent_pid = 4100
    child_pid = 4101
    supervisor = {
        "role": "autodiscover_supervisor",
        "pid": supervisor_pid,
        "ppid": 2000,
        "pgid": supervisor_pid,
        "sid": supervisor_pid,
        "started_at": 99.0,
        "argv": verifier["proxy_supervisor_argv"](),
    }
    parent = {
        "role": "uv_parent",
        "pid": parent_pid,
        "ppid": supervisor_pid,
        "pgid": parent_pid,
        "sid": parent_pid,
        "started_at": 100.0,
        "argv": verifier["proxy_parent_argv"]("dynamic"),
    }
    child = {
        "role": "listener_child",
        "pid": child_pid,
        "ppid": parent_pid,
        "pgid": parent_pid,
        "sid": parent_pid,
        "started_at": 101.0,
        "argv": verifier["proxy_child_argv"]("dynamic"),
    }
    runtime = {
        "proxy_status": {
            "stdout": verifier["canonical_proxy_status"](parent_pid, supervisor_pid),
            "returncode": 0,
            "argv": ["sparkrun", "proxy", "status"],
        },
        "proxy_processes": [supervisor, parent, child],
        "proxy_listener": {
            "bind": "0.0.0.0",
            "port": 4000,
            "inode": "1234",
            **child,
        },
        "proxy_upstream_sockets": [{
            "owner_pid": child_pid,
            "state": "ESTABLISHED",
            "remote_address": "192.168.178.47",
            "remote_port": 8000,
            "inode": "5678",
        }],
    }
    validate = verifier["proxy_binding_failures"]
    assert validate(runtime) == []
    mutations = {
        "substituted status parent": lambda row: row["proxy_status"].__setitem__(
            "stdout", verifier["canonical_proxy_status"](9999, supervisor_pid)
        ),
        "substituted status supervisor": lambda row: row["proxy_status"].__setitem__(
            "stdout", verifier["canonical_proxy_status"](parent_pid, 9998)
        ),
        "substituted listener": lambda row: row["proxy_listener"].__setitem__("pid", 9999),
        "stale extra descendant": lambda row: row["proxy_processes"].append(
            dict(child, pid=4999, ppid=parent_pid, started_at=99.0)
        ),
        "stale extra supervisor": lambda row: row["proxy_processes"].append(
            dict(supervisor, pid=4998, started_at=98.0)
        ),
        "missing upstream socket": lambda row: row.__setitem__("proxy_upstream_sockets", []),
    }
    for name, mutate in mutations.items():
        forged = json.loads(json.dumps(runtime))
        mutate(forged)
        assert validate(forged), name
    capture = (EVIDENCE / "capture_runtime.py").read_text()
    assert "/proc" in capture and "proxy_upstream_sockets" in capture
    assert "shell=True" not in capture
    controls = (EVIDENCE / "test_negative_controls.py").read_text()
    for control in (
        "proxy status PID substituted",
        "proxy status autodiscovery supervisor PID substituted",
        "proxy listener identity substituted",
        "stale extra LiteLLM descendant",
        "stale extra autodiscovery supervisor",
        "proxy upstream socket removed",
    ):
        assert control in controls


def test_telemetry_parser_enforces_complete_ordered_grammar_and_cardinality(
    tmp_path: Path,
) -> None:
    verifier = runpy.run_path(str(EVIDENCE / "verify.py"))
    source = (EVIDENCE / "load-telemetry.log").read_text()
    parsed = verifier["parse_telemetry"](EVIDENCE / "load-telemetry.log")
    for sample in ("before", "during", "after"):
        for host in verifier["EXPECTED_HOST_ORDER"]:
            assert len(parsed[sample]["hosts"][host]["gpu_samples"]) == 10

    incomplete, removed = re.subn(
        r"^P\d+, \d+ %, \d+ MHz, [0-9.]+ W\n", "", source, count=1, flags=re.MULTILINE
    )
    assert removed == 1
    mutations = {
        "unknown": source + "unknown=record\n",
        "duplicate": source.replace(
            "vllm:num_requests_waiting{",
            "vllm:num_requests_running{engine=\"0\",model_name=\"GLM-5.3-Flash-EXL3\"} 0.0\n"
            "vllm:num_requests_waiting{",
            1,
        ),
        "incomplete": incomplete,
        "out-of-order": source.replace(
            "host=192.168.178.47 acquisition_started_at=",
            "host=192.168.178.46 acquisition_started_at=",
            1,
        ),
    }
    for name, text in mutations.items():
        path = tmp_path / f"{name}.log"
        path.write_text(text)
        try:
            verifier["parse_telemetry"](path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"telemetry parser accepted {name} mutation")
    controls = (EVIDENCE / "test_negative_controls.py").read_text()
    for control in (
        "telemetry unknown line",
        "telemetry duplicate record",
        "telemetry incomplete GPU sample set",
        "telemetry out-of-order host",
    ):
        assert control in controls


def test_upstream_compatibility_suite_and_exact_source_parity_gate_are_explicit() -> None:
    compatibility = MOD / "run_upstream_compatibility_suite.sh"
    parity = MOD / "verify_upstream_source_parity.py"
    assert compatibility.is_file() and parity.is_file()
    compatibility_text = compatibility.read_text()
    assert "set -euo pipefail" in compatibility_text
    assert "uv run --with pytest --with torch --with Jinja2 --with numpy" in compatibility_text
    assert "python -m pytest -q" in compatibility_text
    assert "--ignore=upstream/tests/test_apc_per_group_retention.py" in compatibility_text
    assert "python3 /upstream/tests/test_apc_per_group_retention.py" in compatibility_text
    assert "--deselect" not in compatibility_text
    parity_text = parity.read_text()
    assert SOURCE_REVISION in parity_text
    assert "MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks" in parity_text
    assert "relative file set" in parity_text and "byte mismatch" in parity_text
    readme = (MOD / "README.md").read_text()
    assert "upstream compatibility suite" in readme
    assert "114 passed and 13 subtests passed" in readme and "1 failed" not in readme
    assert "all pytest-compatible upstream checks pass" in readme
    assert "verify_upstream_source_parity.py" in readme
    assert "run_upstream_compatibility_suite.sh" in readme


def test_upstream_indexer_wiring_matches_rightsize_default() -> None:
    start = (MOD / "upstream" / "start.sh").read_text()
    upstream_test = (MOD / "upstream" / "tests" / "test_indexer_workspace.py").read_text()
    assert 'GLM53_INDEXER_WORKSPACE="${GLM53_INDEXER_WORKSPACE-rightsize}"' in start
    assert 'GLM53_INDEXER_WORKSPACE="${GLM53_INDEXER_WORKSPACE-rightsize}"' in upstream_test
    assert 'GLM53_INDEXER_WORKSPACE="${GLM53_INDEXER_WORKSPACE-stock}"' not in upstream_test


def test_mod_manifest_covers_every_file_and_hashes_match() -> None:
    manifest = MOD / "SHA256SUMS"
    assert manifest.is_file()
    declared: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        assert rel not in declared
        declared[rel] = digest
    actual = {
        path.relative_to(MOD).as_posix()
        for path in MOD.rglob("*")
        if path.is_file() and path != manifest and "__pycache__" not in path.parts
    }
    assert set(declared) == actual
    for rel, expected in declared.items():
        assert hashlib.sha256((MOD / rel).read_bytes()).hexdigest() == expected, rel
