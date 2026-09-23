#!/usr/bin/env python3
"""Fail closed on the final composed GLM-5.3 runtime, not patch markers alone."""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import os

import sys
from pathlib import Path
from typing import Iterable

MOD = Path(__file__).resolve().parent
OVERLAY = MOD / "upstream/overlay"
DEFAULT_SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
E3_EXECUTION_MARKER = "[glm53-e3-executed]"


def load_script(path: Path) -> dict:
    name = f"_verify_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load patch source: {path}")
    module = importlib.util.module_from_spec(spec)
    # inspect.getsource() in fair-v5 needs a real registered module, not the
    # ephemeral globals returned by runpy.run_path().
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return vars(module)


def literal_constants(path: Path, names: set[str]) -> dict[str, object]:
    found = {}
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    if missing := names - found.keys():
        raise RuntimeError(f"{path}: missing literal constants {sorted(missing)}")
    return found


def fragment_failures(label: str, text: str, required: Iterable[str],
                      forbidden: Iterable[str] = ()) -> list[str]:
    required = list(required)
    failures = []
    for index, fragment in enumerate(required):
        count = text.count(fragment)
        if count != 1:
            failures.append(f"{label}: required fragment {index} count={count}, expected=1")
    for index, fragment in enumerate(forbidden):
        # A stock anchor can remain inside its exact replacement. Additional
        # stock copies are still forbidden. Contracts use maximal nonoverlapping
        # replacements, so nested snippets cannot double-count that allowance.
        expected = sum(replacement.count(fragment) for replacement in required)
        count = text.count(fragment)
        if count != expected:
            failures.append(f"{label}: forbidden fragment {index} count={count}, expected={expected}")
    return failures


def check_file(path: Path, label: str, required: Iterable[str],
               forbidden: Iterable[str] = (), *, compile_python: bool = True) -> list[str]:
    if not path.is_file():
        return [f"{label}: missing {path}"]
    text = path.read_text()
    failures = fragment_failures(label, text, required, forbidden)
    if compile_python:
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"{label}: syntax error: {exc}")
    return failures


def composed_contracts() -> list[tuple[str, str, list[str], list[str]]]:
    """Canonical final snippets, composed in the runtime's reviewed patch order.

    Patches that touch the same file are checked together. Later replacements
    are explicitly applied to earlier owned regions (notably compact drafter
    geometry). Never treat marker presence or installer 'skipped' as a pass.
    """
    rows: dict[str, tuple[list[str], list[str], list[str]]] = {}

    def add(path, label, required, forbidden=()):
        labels, req, old = rows.setdefault(path, ([], [], []))
        labels.append(label)
        req.extend(required)
        old.extend(forbidden)

    def edits(path, label, pairs):
        add(path, label, [new for _, old, new in pairs], [old for _, old, new in pairs])

    def script(name):
        return load_script(OVERLAY / f"patch_{name}.py")

    stop = script("suppress_stops_in_reasoning")
    local = load_script(MOD / "patch_suppress_stops_multitoken.py")
    add("v1/engine/detokenizer.py", "reasoning stop policy",
        [stop["IMPORT_NEW"], stop["INIT_NEW"], local["NEW"], local["CHECK_NEW"]],
        [stop["FACTORY_OLD"], stop["INIT_OLD"], stop["STOP_OLD"], local["OLD"],
         local["BAD_NEW"], local["CHECK_OLD"], local["CHECK_BAD"]])

    floor = script("scheduler_decode_floor")
    adaptive = script("adaptive_k")
    scheduler = "v1/core/sched/scheduler.py"
    add(scheduler, "fair scheduler v5",
        [floor["IMPORT_NEW"], floor["_helper_text"]()] + [new for new, old, label in floor["V5_PAIRS"]],
        [old for new, old, label in floor["V5_PAIRS"]])
    edits(scheduler, "Mamba chunk alignment", script("mamba_align_chunking")["EDITS"])
    add(scheduler, "adaptive-k",
        [adaptive[k] for k in ("SCHED_HELPER", "OBS_NEW", "UPD_NEW", "SCHED_K_NEW")],
        [adaptive[k] for k in ("OBS_OLD", "UPD_OLD", "SCHED_K_OLD")])
    add("v1/worker/gpu/cudagraph_utils.py", "adaptive-k cudagraph",
        [adaptive["CG_HELPER"], adaptive["CG_NEW"]], [adaptive["CG_OLD"]])

    hybrid = script("hybrid_prefix_hit")
    group = script("apc_per_group_retention")
    coordinator = "v1/core/kv_cache_coordinator.py"
    # Latest hybrid boundary stage preserves INIT_NEW but replaces MIN_NEW's
    # verify preamble. Retention reuses BASE_HELPER and the plain os import.
    add(coordinator, "hybrid APC boundary/replay",
        [hybrid["IMPORT_NEW"]] + [text for _, text in hybrid["REQUIRED_ONCE"]],
        [text for _, text in hybrid["SUPERSEDED"]] + [hybrid["LOG_OLD"], hybrid["INIT_OLD"]])
    add(coordinator, "per-group retention",
        [group[k] for k in ("RETENTION_HELPER", "DFLASH_PRIOR_HELPER", "INIT_FINAL",
                           "BASE_CACHE_NEW", "HYBRID_LOOP_NEW", "HYBRID_CACHE_NEW",
                           "FREE_METHOD_NEW", "REMOVE_SKIPPED_NEW")],
        [group[k] for k in ("INIT_OLD", "BASE_CACHE_OLD", "HYBRID_LOOP_OLD",
                           "HYBRID_CACHE_OLD", "FREE_OLD", "REMOVE_SKIPPED_OLD")])
    add("v1/core/block_pool.py", "APC block priority",
        [group["BP_INIT_NEW"], group["BP_FREE_NEW"]], [group["BP_INIT_OLD"], group["BP_FREE_OLD"]])
    manager = "v1/core/single_type_kv_cache_manager.py"
    add(manager, "DFlash prior retention", [group["STM_REACHABLE_NEW"]], [group["STM_REACHABLE_OLD"]])

    nostore = script("apc_no_store")
    paths = {"sampling_params.py": "sampling_params.py", "request.py": "v1/request.py",
             "block_pool.py": "v1/core/block_pool.py"}
    for name, (_, plan, requires) in nostore["PLAN"].items():
        edits(paths[name], "APC no-store", plan)
        add(paths[name], "no-store prerequisites", requires)
    mamba = script("mamba_align_state_free")
    edits(manager, "Mamba state free", mamba["MANAGER_EDITS"])
    edits("v1/kv_cache_interface.py", "Mamba state capacity", mamba["SPEC_EDITS"])

    draft = script("glm5_drafter_group")
    compact_pairs = [(draft[f"COMPACT_{name}_OLD"], draft[f"COMPACT_{name}_NEW"])
                     for name in ("SELECTION", "PREFLIGHT", "VALIDATION")]
    final_draft = []
    for _, old, new in draft["EDITS"]:
        for compact_old, compact_new in compact_pairs:
            new = new.replace(compact_old, compact_new)
        # The public image's already-patched grouping preserves two historical
        # comment paragraphs. Upstream migrates their code but not comments.
        # Split only those paragraphs: every executable byte remains required.
        for comment in (
            "    # group ids stay stable. Exact-fit pages permit virtual splitting;\n"
            "    # padded pages require matching manager and kernel block sizes.\n",
            "        # Padded slot-sharing requires an unsplit manager block;\n"
            "        # worker-side kernel selection checks the actual backend.\n",
        ):
            new = new.replace(comment, "\0")
        final_draft.extend(new.split("\0"))
    add("v1/core/kv_cache_utils.py", "DFlash group/compact-off geometry",
        final_draft + [draft["COMPACT_BLOCK_HELPER"]] + [new for old, new in compact_pairs],
        [old for _, old, new in draft["EDITS"]] + [old for old, new in compact_pairs])
    add("v1/worker/utils.py", "DFlash padded-page kernel guard",
        ["    MambaSpec,\n    SlidingWindowSpec,\n    UniformTypeKVCacheSpecs,", draft["KERNEL_GUARD_NEW"]],
        ["    MambaSpec,\n    UniformTypeKVCacheSpecs,", draft["KERNEL_GUARD_OLD"]])
    capacity = script("kv_capacity_log")
    add("v1/core/kv_cache_utils.py", "KV capacity log",
        [capacity["PATCHED_HELPERS"], capacity["PATCHED_CALL"]])
    tool = script("tool_choice_none")
    add("parser/glm47_moe.py", "tool_choice none", [tool["NEW"], 'TOOL_CALL_START = "<tool_call>"'], [tool["OLD"]])
    default = script("default_max_new_tokens")
    for path_key, prefix in (("LIMITS_PATH", ""), ("COMPLETION_PATH", "COMPLETION_"), ("PROTOCOL_PATH", "PROTOCOL_")):
        add(str(default[path_key]), "omitted-only output default", [default[prefix + "NEW"]], [default[prefix + "OLD"]])
    cache = script("cache_reset")
    add("entrypoints/openai/api_server.py", "cache reset opt-in", [cache["DEV_NEW"]], [cache["DEV_OLD"]])

    result = []
    for path, (labels, required, forbidden) in rows.items():
        # Nested canonical fragments are already covered by their enclosing
        # replacement. Deduplicate before computing embedded old allowances.
        unique = list(dict.fromkeys(required))
        maximal = [item for item in unique if not any(item != other and item in other for other in unique)]
        result.append((path, "; ".join(labels), maximal, list(dict.fromkeys(forbidden))))
    return result


def verify_video(site: Path, *, import_runtime: bool = True) -> list[str]:
    failures = []
    source = OVERLAY / "patch_glm_video_placeholders.py"
    installed = site.parent / "glm53_video_patch.py"
    pth = site.parent / "glm53_video.pth"
    if not installed.is_file() or installed.read_bytes() != source.read_bytes():
        failures.append("video: installed patch bytes do not equal vendored source")
    if not pth.is_file() or pth.read_text() != "import glm53_video_patch\n":
        failures.append("video: .pth import hook missing or altered")
    constants = literal_constants(source, {"KPOOL_OLD", "KPOOL_NEW"})
    failures.extend(check_file(site / "model_executor/layers/sparse_attn_indexer_kpool.py",
                               "video persistent-topk", [str(constants["KPOOL_NEW"])], [str(constants["KPOOL_OLD"])]))
    if not import_runtime:
        return failures  # CPU extracted-image tests only; CLI never disables this.
    try:
        module = importlib.import_module("vllm.model_executor.models.glm4_1v")
        cls = module.Glm4vProcessingInfo
        method = cls._construct_video_placeholder
        if getattr(cls, "_glm53_video_t_aligned", False) is not True:
            failures.append("video: Glm4vProcessingInfo alignment marker is not true")
        if method.__module__ != "glm53_video_patch":
            failures.append(f"video: placeholder method owner is {method.__module__!r}")
        for name in ("_align_timestamps", "_get_video_second_idx_glm46v"):
            if name not in method.__code__.co_names:
                failures.append(f"video: placeholder method missing {name}")
    except Exception as exc:
        failures.append(f"video: live import/state check failed: {exc!r}")
    return failures


def verify_runtime(site: Path, *, payload_root: Path = Path("/opt/glm53"),
                   import_runtime: bool = True) -> list[str]:
    failures = []
    for relative, label, required, forbidden in composed_contracts():
        failures.extend(check_file(site / relative, label, required, forbidden))

    # These upstream verifiers also check owned helper binding counts, so a
    # duplicate definition or late reassignment cannot hide behind exact bytes.
    hybrid = load_script(OVERLAY / "patch_hybrid_prefix_hit.py")
    coordinator = site / "v1/core/kv_cache_coordinator.py"
    if coordinator.is_file():
        failures.extend(f"hybrid APC: {error}" for error in hybrid["verify_complete"](coordinator.read_text()))
    for filename, relative, function in (
        ("patch_kpool_tail_slotmap.py", "v1/worker/block_table.py", "verified_state"),
        ("patch_indexer_workspace.py", "v1/attention/backends/mla/indexer.py", "verified_state"),
        ("patch_kv_capacity_log.py", "v1/core/kv_cache_utils.py", "verified_state"),
        ("patch_xgrammar_termination.py", "v1/structured_output/backend_xgrammar.py", "verified_backend_state"),
        ("patch_xgrammar_termination.py", "v1/structured_output/__init__.py", "verified_manager_state"),
    ):
        ns = load_script(OVERLAY / filename)
        target = site / relative
        if not target.is_file() or not ns[function](target.read_text()):
            failures.append(f"{filename}: exact patched state missing: {relative}")
        elif target.is_file():
            failures.extend(check_file(target, filename, []))

    spin = load_script(OVERLAY / "patch_spinwait.py")
    spin_path = site / "distributed/device_communicators/shm_broadcast.py"
    if not spin_path.is_file():
        failures.append(f"spinwait: missing {spin_path}")
    else:
        try:
            patched, _ = spin["prepare"](spin_path.read_text(), None)
            if patched != spin_path.read_text():
                failures.append("spinwait: stock profile would alter runtime source")
        except ValueError as exc:
            failures.append(f"spinwait: {exc}")

    dense = load_script(OVERLAY / "patch_dense_fp8.py")
    marker = load_script(MOD / "patch_e3_execution_marker.py")
    if E3_EXECUTION_MARKER not in marker["HELPER"]:
        failures.append("E3 marker overlay: expected runtime marker missing")
    source = payload_root / "exl3.py"
    target = site / "model_executor/layers/quantization/exl3.py"
    for path in (source, target):
        if not path.is_file():
            failures.append(f"E3 marker: missing {path}")
            continue
        try:
            same, status = marker["apply_text"](path.read_text())
            if status != "skipped" or same != path.read_text():
                failures.append(f"E3 marker: exact marker state missing: {path}")
        except ValueError as exc:
            failures.append(f"E3 marker: {exc}")
    if not source.is_file() or not target.is_file() or source.read_bytes() != target.read_bytes():
        failures.append("dense-fp8: EXL3 overlay bytes are not installed exactly")
    if os.environ.get("GLM53_DENSE_FP8", "off").strip().lower() not in ("", "off", "0", "no", "none"):
        failures.append("dense-fp8: immutable profile must remain off")
    kda = site / "models/glm5next/nvidia/kda.py"
    if not kda.is_file():
        kda = site / "model_executor/models/glm5next/nvidia/kda.py"
    failures.extend(check_file(kda, "dense-fp8 KDA off-state", [dense["KDA_OLD"]], [dense["KDA_NEW"]]))
    failures.extend(check_file(kda.parent / "model.py", "dense-fp8 MLA off-state", [dense["MLA_OLD"]], [dense["MLA_NEW"]]))
    ablit = load_script(MOD / "patch_ablit.py")
    nvidia = site / "models/glm5next/nvidia"
    helper_source = payload_root / "ablit_runtime.py"
    helper_target = nvidia / "glm53_ablit.py"
    if not helper_source.is_file() or not helper_target.is_file() or helper_source.read_bytes() != helper_target.read_bytes():
        failures.append("ABLIT: runtime helper bytes are not installed exactly")
    for name in ("model.py", "mtp.py"):
        failures.extend(check_file(nvidia / name, "ABLIT model hook", [ablit["HOOK_LINES"]]))
    failures.extend(verify_video(site, import_runtime=import_runtime))
    return failures


def self_test() -> int:
    contracts = composed_contracts()
    marker = load_script(MOD / "patch_e3_execution_marker.py")
    contracts.append(("exl3.py", "E3 runtime execution marker", [marker["HELPER"], marker["CALL_NEW"]], [marker["CALL_OLD"]]))
    mutations = 0
    embedded_forbidden = 0
    for path, label, required, forbidden in contracts:
        clean = "\n".join(required)
        if errors := fragment_failures(label, clean, required, forbidden):
            raise AssertionError(f"self-test positive control: {errors}")
        for fragment in required:
            for changed in (clean.replace(fragment, "", 1), clean + fragment):
                if not fragment_failures(label, changed, required, forbidden):
                    raise AssertionError(f"self-test accepted missing/duplicate block: {label}")
                mutations += 1
        for fragment in forbidden:
            if not fragment_failures(label, clean + fragment, required, forbidden):
                raise AssertionError(f"self-test accepted stale block: {label}")
            embedded_forbidden += int(any(fragment in item for item in required))
            mutations += 1
    print("runtime patch verifier self-test: PASS "
          f"contracts={len(contracts)} embedded_forbidden={embedded_forbidden} mutations={mutations}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    failures = verify_runtime(args.site.resolve())
    if failures:
        for failure in failures:
            print(f"FATAL: {failure}", file=sys.stderr)
        return 1
    print("[OK] complete GLM-5.3 runtime patched state verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
