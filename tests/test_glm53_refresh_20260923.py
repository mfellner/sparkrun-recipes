"""CPU-only contracts for the 3f2be18 TP2 refresh (not live acceptance)."""
import hashlib
import json
import re
import shlex
import subprocess
import sys
import runpy

import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mods/glm-5.3-flash-exl3-upstream-850k"
RECIPE = ROOT / "recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml"
REVISION = "3f2be18c41effca0b4b2c6a65f0b24a7a9f39567"


def test_complete_upstream_tree_matches_exact_reviewed_commit():
    # Canonical sorted path/content digest calculated from `git archive` of the
    # reviewed commit, independent of the writable mod manifest.
    rows = {p.relative_to(MOD / "upstream").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (MOD / "upstream").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
    assert len(rows) == 258
    digest = hashlib.sha256("".join(f"{rows[n]}  {n}\n" for n in sorted(rows)).encode()).hexdigest()
    assert digest == "990b11956a7c5876f6eae60316db4232df7379f9c701cf6472a49b1f3f433d89"
    assert REVISION in (MOD / "verify_upstream_source_parity.py").read_text()


def test_latest_tp2_defaults_preserve_local_boundaries():
    recipe = yaml.safe_load(RECIPE.read_text())
    assert recipe["metadata"]["source_revision"] == REVISION
    assert recipe["container"] == "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d"
    env = recipe["env"]
    expected = {
        "DEFAULT_MAX_NEW_TOKENS": "65536",
        "GLM53_MIXED_PREFILL_CHUNK": "fair",
        "GLM53_FAIR_PREFILL_CHUNK": "256",
        "GLM53_FAIR_PREFILL_SHARE": "0.30",
        "GLM53_FAIR_PREFILL_MAX_INTERVAL_MS": "2000",
        "GLM53_FAIR_PREFILL_MAX_STEP_MS": "2000",
        "GLM53_FAIR_PREFILL_MAX_CHUNKS": "1",
        "GLM53_APC_NO_STORE": "1", "GLM53_KV_CAPACITY_LOG": "1",
        "GLM53_EXPOSE_CACHE_RESET": "0", "GLM53_DRAFT_KV_COMPACT": "0",
        "GLM53_EXL3_MOE_FAST": "0", "GLM53_KDA_BF16_LARGE_M": "0",
        "GLM53_COOP_GEOMETRY": "", "GLM53_ADAPTIVE_K": "off",
        "GLM53_DENSE_FP8": "off", "GLM53_SPINWAIT_MS": "stock",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "SPT_NOENV": "1",  # Keep TP-worker /proc environment auditable after setproctitle.
    }
    for key, value in expected.items():
        assert env.get(key) == value, key
    argv = shlex.split(recipe["command"].replace("\\\n", " "))
    for key, value in {
        "--load-format": "instanttensor", "--kv-cache-memory-bytes": "15032385536",
        "--mm-processor-cache-gb": "1", "--port": "8000",
        "--max-model-len": "850000", "--max-num-batched-tokens": "7168",
        "--served-model-name": "GLM-5.3-Flash-EXL3",
        "--allowed-media-domains": "media.invalid",
    }.items():
        assert argv.count(key) == 1
        assert argv[argv.index(key) + 1] == value
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1]) == {"max_image_tokens": 2048}
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {"image": 4, "video": 0}
    assert "--media-io-kwargs" not in argv  # upstream frame override unset; local video disabled
    assert "--override-generation-config" not in argv  # omitted-only, never an explicit client cap
    assert argv.count("--enable-prompt-tokens-details") == 1  # per-request APC evidence


def test_runtime_order_matches_upstream_with_only_reviewed_local_insertions():
    source = (MOD / "upstream/start.sh").read_text()
    expected = re.search(r"GLM53_OVERLAY_ORDER=\((.*?)\n\)", source, re.S).group(1).split()
    run = (MOD / "run.sh").read_text()
    actual = re.findall(r"^python3 (?:upstream/overlay/)?(patch_\w+\.py)$", run, re.M)
    assert actual.count("patch_suppress_stops_multitoken.py") == 1
    assert actual.index("patch_suppress_stops_multitoken.py") == actual.index("patch_suppress_stops_in_reasoning.py") + 1
    assert actual.index("patch_e3_execution_marker.py") == actual.index("patch_dense_fp8.py") - 1
    assert [p for p in actual if p not in ("patch_suppress_stops_multitoken.py", "patch_e3_execution_marker.py")] == expected
    assert REVISION in run
    assert run.index("python3 verify_runtime_patch_state.py") > run.index("python3 patch_ablit.py")


def test_composed_verifier_selftest_covers_new_runtime_surfaces():
    ns = runpy.run_path(str(MOD / "verify_runtime_patch_state.py"))
    assert "composed_contracts" in ns, "verifier must derive final composed, not intermediate snippets"
    contracts = ns["composed_contracts"]()
    assert {str(row[0]) for row in contracts} >= {
        "v1/core/sched/scheduler.py", "v1/core/kv_cache_utils.py", "v1/worker/utils.py",
        "v1/core/kv_cache_coordinator.py", "v1/core/single_type_kv_cache_manager.py",
        "v1/kv_cache_interface.py", "sampling_params.py", "v1/request.py",
        "v1/core/block_pool.py", "parser/glm47_moe.py",
        "entrypoints/serve/utils/api_utils.py", "entrypoints/openai/completion/serving.py",
        "entrypoints/openai/completion/protocol.py", "entrypoints/openai/api_server.py",
    }
    for path, label, required, forbidden in contracts:
        assert required and all(isinstance(item, str) and item for item in required), label
        clean = "\n".join(required)
        assert ns["fragment_failures"](label, clean, required, forbidden) == [], label
        for fragment in required:
            assert ns["fragment_failures"](label, clean.replace(fragment, "", 1), required, forbidden), label
            assert ns["fragment_failures"](label, clean + fragment, required, forbidden), label
            middle = len(fragment) // 2
            drifted = fragment[:middle] + "DRIFT" + fragment[middle:]
            assert ns["fragment_failures"](label, clean.replace(fragment, drifted, 1), required, forbidden), label
        for fragment in forbidden:
            assert ns["fragment_failures"](label, clean + fragment, required, forbidden), label
    assert ns["self_test"]() == 0


def test_composed_verifier_against_real_cpu_patched_image(monkeypatch):
    # Opt-in extracted disposable-image fixture; never import the runtime/GPU.
    import os
    import pytest
    root = os.environ.get("GLM53_PATCHED_FIXTURE")
    if not root:
        pytest.skip("requires CPU-patched exact-image fixture (not live serving)")
    root = Path(root)
    ns = runpy.run_path(str(MOD / "verify_runtime_patch_state.py"))
    failures = ns["verify_runtime"](root / "site/vllm", payload_root=root / "payload", import_runtime=False)
    assert failures == [], failures
    # Exercise the SAME final-file contracts on real patched sources, not only
    # synthetic marker strings. No production file is written by mutations.
    controls = 0
    for relative, label, required, forbidden in ns["composed_contracts"]():
        text = (root / "site/vllm" / relative).read_text()
        assert ns["fragment_failures"](label, text, required, forbidden) == []
        for fragment in required:
            assert ns["fragment_failures"](label, text.replace(fragment, "", 1), required, forbidden)
            assert ns["fragment_failures"](label, text + fragment, required, forbidden)
            controls += 2
        for fragment in forbidden:
            assert ns["fragment_failures"](label, text + fragment, required, forbidden)
            controls += 1
    assert controls > 200


def test_instanttensor_loader_is_checked_before_serving():
    run = (MOD / "run.sh").read_text()
    assert 'metadata.version(\"instanttensor\") == \"0.2.0\"' in run
    assert run.index('metadata.version(\"instanttensor\")') < run.index('import torch, exllamav3_ext')


def test_latest_warmup_is_wired_under_existing_fail_closed_supervision():
    upstream = (MOD / "upstream/scripts/boot-shape-warmup.sh").read_text()
    gate = (MOD / "postready_gate.sh").read_text()
    assert "PREFILL_S=(3584 7168 14336 65536)" in upstream
    assert 'prefill\n' in upstream
    assert 'GLM53_WARMUP_MAX_CONCURRENCY=4' in gate
    assert '"$MOD_DIR/upstream/scripts/boot-shape-warmup.sh" "$BASE" "$MODEL" || fail 92' in gate
    assert 'threading.Barrier(4)' in gate
    assert 'kill -TERM -- "-$SERVE_PGID"' in gate
    assert 'filler="alpha "*55000' in gate


def test_compatibility_runner_cannot_use_loaded_host_gpus():
    runner = (MOD / "run_upstream_compatibility_suite.sh").read_text()
    assert "--network none" in runner and "--memory" in runner
    assert "--gpus" not in runner
    assert "GLM53_TEST_MEMORY" in runner
    assert "GLM53_REQUIRE_COMPOSITION=1" in runner
    assert "GLM53_REQUIRE_VLLM=1" in runner
    assert "test_exl3_thin_fast_gpu.py" in runner
    assert "test_kda_bf16_large_m_gpu.py" in runner
    assert "GLM53_VLLM_SRC" in runner


def test_cpu_composition_harness_refuses_unscoped_execution():
    script = MOD / "test_runtime_patch_composition.py"
    assert script.is_file()
    result = subprocess.run([sys.executable, "-S", str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "disposable CPU-only container" in result.stderr


def test_mod_manifest_covers_every_artifact_once():
    lines = (MOD / "SHA256SUMS").read_text().splitlines()
    rows = [line.split("  ", 1) for line in lines]
    names = [name for digest, name in rows]
    actual = {p.relative_to(MOD).as_posix() for p in MOD.rglob("*") if p.is_file()
              and p.name != "SHA256SUMS" and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
    assert len(names) == len(set(names))
    assert set(names) == actual
    for digest, name in rows:
        assert hashlib.sha256((MOD / name).read_bytes()).hexdigest() == digest, name
