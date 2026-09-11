#!/usr/bin/env python3
"""Fail closed unless every GLM-5.3 runtime patch is complete."""
from __future__ import annotations

import argparse
import ast
import importlib
import os
import runpy
import sys
from pathlib import Path
from typing import Iterable

MOD = Path(__file__).resolve().parent
OVERLAY = MOD / "upstream/overlay"
DEFAULT_SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
E3_EXECUTION_MARKER = "[glm53-e3-executed]"


def load_script(path: Path) -> dict:
    return runpy.run_path(str(path), run_name=f"_verify_{path.stem}")


def literal_constants(path: Path, names: set[str]) -> dict[str, object]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in names:
            found[target.id] = ast.literal_eval(node.value)
    missing = names - found.keys()
    if missing:
        raise RuntimeError(f"{path}: missing literal constants {sorted(missing)}")
    return found


def fragment_failures(
    label: str,
    text: str,
    required: Iterable[str],
    forbidden: Iterable[str] = (),
) -> list[str]:
    required = list(required)
    failures: list[str] = []
    for index, fragment in enumerate(required):
        count = text.count(fragment)
        if count != 1:
            failures.append(f"{label}: required fragment {index} count={count}, expected=1")
    for index, fragment in enumerate(forbidden):
        # A stock fragment may legitimately occur inside one exact required
        # replacement. Count those embedded occurrences instead of skipping the
        # stock check, so an appended mixed required+stock state fails closed.
        expected = sum(replacement.count(fragment) for replacement in required)
        count = text.count(fragment)
        if count != expected:
            failures.append(
                f"{label}: forbidden fragment {index} count={count}, expected={expected}"
            )
    return failures


def check_file(
    path: Path,
    label: str,
    required: Iterable[str],
    forbidden: Iterable[str] = (),
    *,
    compile_python: bool = True,
) -> list[str]:
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


def verify_video(site: Path) -> list[str]:
    failures: list[str] = []
    dist = site.parent
    source = OVERLAY / "patch_glm_video_placeholders.py"
    installed = dist / "glm53_video_patch.py"
    pth = dist / "glm53_video.pth"
    if not installed.is_file() or installed.read_bytes() != source.read_bytes():
        failures.append("video: installed patch bytes do not equal vendored source")
    if not pth.is_file() or pth.read_text() != "import glm53_video_patch\n":
        failures.append("video: .pth import hook missing or altered")

    constants = literal_constants(source, {"KPOOL_OLD", "KPOOL_NEW"})
    failures.extend(
        check_file(
            site / "model_executor/layers/sparse_attn_indexer_kpool.py",
            "video persistent-topk",
            [str(constants["KPOOL_NEW"])],
            [str(constants["KPOOL_OLD"])],
        )
    )
    try:
        module = importlib.import_module("vllm.model_executor.models.glm4_1v")
        cls = module.Glm4vProcessingInfo
        method = cls._construct_video_placeholder
        if getattr(cls, "_glm53_video_t_aligned", False) is not True:
            failures.append("video: Glm4vProcessingInfo alignment marker is not true")
        if method.__module__ != "glm53_video_patch":
            failures.append(f"video: placeholder method owner is {method.__module__!r}")
        names = set(method.__code__.co_names)
        for name in ("_align_timestamps", "_get_video_second_idx_glm46v"):
            if name not in names:
                failures.append(f"video: placeholder method missing {name}")
    except Exception as exc:  # noqa: BLE001 - gate must report every failure
        failures.append(f"video: live import/state check failed: {exc!r}")
    return failures


def verify_runtime(site: Path) -> list[str]:
    failures: list[str] = []

    stop = load_script(OVERLAY / "patch_suppress_stops_in_reasoning.py")
    local_stop = load_script(MOD / "patch_suppress_stops_multitoken.py")
    detok = site / "v1/engine/detokenizer.py"
    failures.extend(
        check_file(
            detok,
            "reasoning stop policy",
            [
                stop["IMPORT_NEW"],
                stop["INIT_NEW"],
                local_stop["NEW"],
                local_stop["CHECK_NEW"],
            ],
            [stop["FACTORY_OLD"], stop["INIT_OLD"], stop["STOP_OLD"], local_stop["OLD"], local_stop["BAD_NEW"], local_stop["CHECK_OLD"], local_stop["CHECK_BAD"]],
        )
    )

    floor = load_script(OVERLAY / "patch_scheduler_decode_floor.py")
    adaptive = load_script(OVERLAY / "patch_adaptive_k.py")
    scheduler = site / "v1/core/sched/scheduler.py"
    failures.extend(
        check_file(
            scheduler,
            "scheduler overlays",
            [
                floor["HELPER"],
                floor["RUNNING_NEW"],
                floor["WAITING_NEW"],
                adaptive["SCHED_HELPER"],
                adaptive["OBS_NEW"],
                adaptive["UPD_NEW"],
                adaptive["SCHED_K_NEW"],
            ],
            [
                floor["RUNNING_OLD"],
                floor["WAITING_OLD"],
                adaptive["OBS_OLD"],
                adaptive["UPD_OLD"],
                adaptive["SCHED_K_OLD"],
            ],
        )
    )
    failures.extend(
        check_file(
            site / "v1/worker/gpu/cudagraph_utils.py",
            "adaptive-k cudagraph",
            [adaptive["CG_HELPER"], adaptive["CG_NEW"]],
            [adaptive["CG_OLD"]],
        )
    )

    hybrid = load_script(OVERLAY / "patch_hybrid_prefix_hit.py")
    failures.extend(
        check_file(
            site / "v1/core/kv_cache_coordinator.py",
            "hybrid APC",
            [hybrid["HELPER"], hybrid["EAGLE_NEW"], hybrid["MIN_NEW"], hybrid["LOG_NEW"]],
            [hybrid["EAGLE_OLD"], hybrid["MIN_OLD"], hybrid["LOG_OLD"]],
        )
    )

    drafter = load_script(OVERLAY / "patch_glm5_drafter_group.py")
    failures.extend(
        check_file(
            site / "v1/core/kv_cache_utils.py",
            "DFlash2 drafter group",
            [replacement for _name, _anchor, replacement in drafter["EDITS"]]
            + [
                "padded slot-share block=%d",
                "s.block_size != 64 or s.page_size_padded != mla_page",
            ],
            [
                "            compact_block = 64\n            if any_draft.block_size > compact_block:\n",
                "            # STANDALONE: the drafter's geometry cannot exactly fill the MLA\n",
            ],
        )
    )

    kpool = load_script(OVERLAY / "patch_kpool_tail_slotmap.py")
    kpool_path = site / "v1/worker/block_table.py"
    if not kpool_path.is_file() or not kpool["verified_state"](kpool_path.read_text() if kpool_path.is_file() else ""):
        failures.append("kpool tail slot-map: exact patched state missing")

    indexer = load_script(OVERLAY / "patch_indexer_workspace.py")
    indexer_path = site / "v1/attention/backends/mla/indexer.py"
    if not indexer_path.is_file() or not indexer["verified_state"](indexer_path.read_text() if indexer_path.is_file() else ""):
        failures.append("indexer workspace: exact patched state missing")

    xgrammar = load_script(OVERLAY / "patch_xgrammar_termination.py")
    backend = site / "v1/structured_output/backend_xgrammar.py"
    manager = site / "v1/structured_output/__init__.py"
    if not backend.is_file() or not xgrammar["verified_backend_state"](backend.read_text() if backend.is_file() else ""):
        failures.append("xgrammar backend: exact termination patch missing")
    if not manager.is_file() or not xgrammar["verified_manager_state"](manager.read_text() if manager.is_file() else ""):
        failures.append("xgrammar manager: exact reasoning patch missing")

    spin = load_script(OVERLAY / "patch_spinwait.py")
    spin_path = site / "distributed/device_communicators/shm_broadcast.py"
    if not spin_path.is_file():
        failures.append(f"spinwait: missing {spin_path}")
    else:
        try:
            patched, _action = spin["prepare"](spin_path.read_text(), None)
            if patched != spin_path.read_text():
                failures.append("spinwait: stock profile would alter runtime source")
        except ValueError as exc:
            failures.append(f"spinwait: {exc}")

    dense = load_script(OVERLAY / "patch_dense_fp8.py")
    e3_marker = load_script(MOD / "patch_e3_execution_marker.py")
    if E3_EXECUTION_MARKER not in e3_marker["HELPER"]:
        failures.append("E3 marker overlay: expected runtime marker missing")
    exl3_source = Path("/opt/glm53/exl3.py")
    exl3_target = site / "model_executor/layers/quantization/exl3.py"
    for marker_path, marker_label in (
        (exl3_source, "E3 marker source"),
        (exl3_target, "E3 marker runtime"),
    ):
        if not marker_path.is_file():
            failures.append(f"{marker_label}: missing {marker_path}")
            continue
        try:
            same, status = e3_marker["apply_text"](marker_path.read_text())
            if status != "skipped" or same != marker_path.read_text():
                failures.append(f"{marker_label}: exact marker state missing")
        except ValueError as exc:
            failures.append(f"{marker_label}: {exc}")
    if not exl3_source.is_file() or not exl3_target.is_file() or exl3_source.read_bytes() != exl3_target.read_bytes():
        failures.append("dense-fp8: EXL3 overlay bytes are not installed exactly")
    if os.environ.get("GLM53_DENSE_FP8", "off").strip().lower() not in ("", "off", "0", "no", "none"):
        failures.append("dense-fp8: immutable profile must remain off")
    kda = site / "models/glm5next/nvidia/kda.py"
    if not kda.is_file():
        kda = site / "model_executor/models/glm5next/nvidia/kda.py"
    model = kda.parent / "model.py"
    failures.extend(check_file(kda, "dense-fp8 KDA off-state", [dense["KDA_OLD"]], [dense["KDA_NEW"]]))
    failures.extend(check_file(model, "dense-fp8 MLA off-state", [dense["MLA_OLD"]], [dense["MLA_NEW"]]))

    ablit = load_script(MOD / "patch_ablit.py")
    nvidia = site / "models/glm5next/nvidia"
    runtime_source = Path("/opt/glm53/ablit_runtime.py")
    runtime_target = nvidia / "glm53_ablit.py"
    if not runtime_source.is_file() or not runtime_target.is_file() or runtime_source.read_bytes() != runtime_target.read_bytes():
        failures.append("ABLIT: runtime helper bytes are not installed exactly")
    failures.extend(check_file(nvidia / "model.py", "ABLIT model hook", [ablit["HOOK_LINES"]]))
    failures.extend(check_file(nvidia / "mtp.py", "ABLIT MTP hook", [ablit["HOOK_LINES"]]))

    failures.extend(verify_video(site))
    return failures


def self_test() -> int:
    stop = load_script(OVERLAY / "patch_suppress_stops_in_reasoning.py")
    local_stop = load_script(MOD / "patch_suppress_stops_multitoken.py")
    floor = load_script(OVERLAY / "patch_scheduler_decode_floor.py")
    adaptive = load_script(OVERLAY / "patch_adaptive_k.py")
    hybrid = load_script(OVERLAY / "patch_hybrid_prefix_hit.py")
    drafter_script = load_script(OVERLAY / "patch_glm5_drafter_group.py")
    dense = load_script(OVERLAY / "patch_dense_fp8.py")
    e3_marker = load_script(MOD / "patch_e3_execution_marker.py")
    video = literal_constants(
        OVERLAY / "patch_glm_video_placeholders.py", {"KPOOL_OLD", "KPOOL_NEW"}
    )
    contracts = [
        (
            "reasoning stop policy",
            [stop["IMPORT_NEW"], stop["INIT_NEW"], local_stop["NEW"], local_stop["CHECK_NEW"]],
            [stop["FACTORY_OLD"], stop["INIT_OLD"], stop["STOP_OLD"], local_stop["OLD"], local_stop["BAD_NEW"], local_stop["CHECK_OLD"], local_stop["CHECK_BAD"]],
        ),
        (
            "scheduler overlays",
            [floor["HELPER"], floor["RUNNING_NEW"], floor["WAITING_NEW"], adaptive["SCHED_HELPER"], adaptive["OBS_NEW"], adaptive["UPD_NEW"], adaptive["SCHED_K_NEW"]],
            [floor["RUNNING_OLD"], floor["WAITING_OLD"], adaptive["OBS_OLD"], adaptive["UPD_OLD"], adaptive["SCHED_K_OLD"]],
        ),
        ("adaptive-k cudagraph", [adaptive["CG_HELPER"], adaptive["CG_NEW"]], [adaptive["CG_OLD"]]),
        (
            "E3 runtime execution marker",
            [e3_marker["HELPER"], e3_marker["CALL_NEW"]],
            [e3_marker["CALL_OLD"]],
        ),
        (
            "hybrid APC",
            [hybrid["HELPER"], hybrid["EAGLE_NEW"], hybrid["MIN_NEW"], hybrid["LOG_NEW"]],
            [hybrid["EAGLE_OLD"], hybrid["MIN_OLD"], hybrid["LOG_OLD"]],
        ),
        (
            "DFlash2 drafter group",
            [replacement for _name, _anchor, replacement in drafter_script["EDITS"]]
            + [
                "padded slot-share block=%d",
                "s.block_size != 64 or s.page_size_padded != mla_page",
            ],
            [
                "            compact_block = 64\n            if any_draft.block_size > compact_block:\n",
                "            # STANDALONE: the drafter's geometry cannot exactly fill the MLA\n",
            ],
        ),
        ("video persistent-topk", [str(video["KPOOL_NEW"])], [str(video["KPOOL_OLD"])]),
        ("dense-fp8 KDA", [dense["KDA_OLD"]], [dense["KDA_NEW"]]),
        ("dense-fp8 MLA", [dense["MLA_OLD"]], [dense["MLA_NEW"]]),
    ]
    mutations = 0
    embedded_forbidden = 0
    for label, required, forbidden in contracts:
        for fragment in required:
            if fragment_failures(label, fragment, [fragment]):
                print(f"self-test positive control failed: {label}", file=sys.stderr)
                return 1
            if not fragment_failures(label, "", [fragment]):
                print(f"self-test accepted missing required fragment: {label}", file=sys.stderr)
                return 1
            mutations += 1
            if not fragment_failures(label, fragment + fragment, [fragment]):
                print(f"self-test accepted duplicate required fragment: {label}", file=sys.stderr)
                return 1
            mutations += 1
        for fragment_index, fragment in enumerate(forbidden):
            containers = [replacement for replacement in required if fragment in replacement]
            if containers:
                clean = "\n".join(containers)
                if fragment_failures(label, clean, containers, [fragment]):
                    print(
                        f"self-test rejected legitimate embedded forbidden fragment: "
                        f"{label}[{fragment_index}]",
                        file=sys.stderr,
                    )
                    return 1
                if not fragment_failures(
                    label, clean + "\n" + fragment, containers, [fragment]
                ):
                    print(
                        f"self-test accepted extra embedded forbidden fragment: "
                        f"{label}[{fragment_index}]",
                        file=sys.stderr,
                    )
                    return 1
                embedded_forbidden += 1
                mutations += 1
                continue
            if fragment_failures(label, "", [], [fragment]):
                print(f"self-test rejected clean forbidden control: {label}", file=sys.stderr)
                return 1
            if not fragment_failures(label, fragment, [], [fragment]):
                print(f"self-test accepted forbidden fragment: {label}", file=sys.stderr)
                return 1
            mutations += 1
    print(
        "runtime patch verifier self-test: PASS "
        f"contracts={len(contracts)} embedded_forbidden={embedded_forbidden} "
        f"mutations={mutations}"
    )
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
