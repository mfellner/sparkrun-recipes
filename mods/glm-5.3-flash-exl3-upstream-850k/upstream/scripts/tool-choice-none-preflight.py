#!/usr/bin/env python3
"""Pre-generation preflight for the tool_choice:"none" live test plan.

Runs the three no-generation gates from docs/tool-choice-none.md inside the
API container (each rank, or head via docker exec):

  A identity   vLLM version, patch state at the import-resolved parser path,
               serving launch flags read from the live process tree.
  B sources    sha256 of every file in the bad_words enforcement chain vs
               the analyzed image state (byte-identical to upstream
               487ecf187d3dfe74d2cf6119a92881dba403c219).
  C tokenizer  <tool_call> vocab/encode equality plus a SamplingParams
               update_from_tokenizer dry run — zero generated tokens.

No requests, no model load, no device work. Phases fail closed: exit 0 only
when every selected phase passes. tokenizer phase needs --model-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

UPSTREAM = "487ecf187d3dfe74d2cf6119a92881dba403c219"
PATCH_MARK = "# [glm53-tool-choice-none]"
OPENER = "<tool_call>"

# Analyzed-image digests (== upstream 487ecf187d3df for every entry).
CHAIN = {
    "vllm/parser/glm47_moe.py":
        "ce3629319e56e882d25cb75d62e3e7088a4eec1518885fc69fc696eafb4a97b2",
    "vllm/parser/abstract_parser.py":
        "e567186750002ed7d0f5c5efeaffc9b9cfbec18060bdc080420b24cade713e13",
    "vllm/parser/engine/parser_engine.py":
        "3ac89a7f22f0e4f0d3f6f2365d79f64da9da217969793f6a33a6db9cf5ef60ff",
    "vllm/parser/engine/adapters.py":
        "dc1c1317dbfb298e54b8d94ca0e66d2b0cb1e481c35cdcc60a815284bd8a6ef7",
    "vllm/entrypoints/openai/chat_completion/serving.py":
        "16ec26bfd4b376ccd2e9f3672a23266cdd9beb91ca82f7423267beaee8ef8a6d",
    "vllm/entrypoints/openai/chat_completion/protocol.py":
        "f16851e236517ddffbdccf40b7c5a6275320033130a8c080f1bf3904b5229453",
    "vllm/entrypoints/generate/base/serving.py":
        "09b0237bd37c7dd719ed04e5b447efbb54941e0c19825e62138c8f6db4d3064e",
    # Deployed image = upstream 487ecf187d3df + the repo's build-time
    # [glm53-apc-no-store] overlay on this one file: three insert-only hunks
    # (helpers, skip_writing_prefix_cache field, one post_init validation
    # call); zero lines touch bad_words/update_from_tokenizer. Owner-approved
    # amended baseline (2026-09-18); the pre-overlay digest was 877c355c…
    "vllm/sampling_params.py":
        "92be3de4deaa968281df6103b736dcaadb5ae09bf39f611e8ca49564d8859adb",
    "vllm/v1/engine/input_processor.py":
        "d468edd32fd32a2d1e1c094695462f525028e097ca8c9a8e4e1c56f32c6a43a0",
    "vllm/v1/sample/sampler.py":
        "315af950ef4c35fced53dc3a5df49a80af20b47e417e8f12bf315f535769bab2",
    "vllm/v1/sample/ops/bad_words.py":
        "0b1d0a9b13b92ddcae2c83393c07a34c0f107813725700dee794caa7fe7f3b12",
    "vllm/v1/sample/rejection_sampler.py":
        "4bb87d7984d967be9c7e5b6a6b489c02571ef2f2913211cb3f1eea74b2e1f32f",
    "vllm/v1/worker/gpu_input_batch.py":
        "0687f0e38bf889cef454a38ee262a5686c1ea459337c42219cb91c900799305f",
}
# glm47_moe.py digest above is the PRE-PATCH image state; the patch rewrites
# it, so phase B accepts either the pre-patch digest or the marker present.


def fail(message: str) -> None:
    print(f"FAIL {message}")


def phase_identity() -> bool:
    ok = True
    try:
        import vllm
        version = getattr(vllm, "__version__", "?")
    except Exception as error:  # noqa: BLE001 - report, then fail
        version = f"import-error: {error}"
    print(f"A: vllm.__version__ = {version}")
    ok &= "g487ecf187" in version
    try:
        import vllm.parser.glm47_moe as module
        source = Path(module.__file__).read_text(encoding="utf-8")
        patched = PATCH_MARK in source
        print(f"A: parser path = {module.__file__}")
        print(f"A: patch       = {'APPLIED' if patched else 'NOT APPLIED'}")
    except Exception as error:  # noqa: BLE001
        fail(f"A: parser import: {error}")
        return False
    flags = ""
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = cmdline.read_bytes().split(b"\0")
        except OSError:
            continue
        if any(b"vllm.entrypoints" in part for part in argv):
            flags = b" ".join(argv).decode(errors="replace")
            break
    if flags:
        for wanted in ("--tool-call-parser glm47", "--enable-auto-tool-choice",
                       "--reasoning-parser glm45"):
            present = wanted in flags
            print(f"A: launch {wanted!r:34} = {'present' if present else 'MISSING'}")
            ok &= present
    else:
        print("A: launch flags = live vLLM process not found (run inside the API container)")
    return ok


def phase_sources() -> bool:
    import vllm
    root = Path(vllm.__file__).resolve().parent
    ok = True
    for name, expected in CHAIN.items():
        path = root / name.removeprefix("vllm/")
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            fail(f"B: {name}: {error}")
            ok = False
            continue
        if digest == expected:
            print(f"B: {name:55} match")
            continue
        if name.endswith("glm47_moe.py") and PATCH_MARK in path.read_text(encoding="utf-8"):
            print(f"B: {name:55} patched (marker present)")
            continue
        fail(f"B: {name}: drift {digest}")
        ok = False
    return ok


def phase_tokenizer(model_dir: str) -> bool:
    # Load the tokenizer the way the engine does: the raw transformers
    # AutoTokenizer on this image returns TokenizersBackend, which lacks
    # max_token_id that update_from_tokenizer reads.
    from vllm.sampling_params import SamplingParams
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(model_dir)
    vocab = tokenizer.get_vocab()
    if OPENER not in vocab:
        fail(f"C: {OPENER!r} not in tokenizer vocab")
        return False
    token_id = vocab[OPENER]
    encoded = tokenizer.encode(OPENER, add_special_tokens=False)
    print(f"C: vocab[{OPENER!r}] = {token_id}; encode = {encoded}")
    if encoded != [token_id]:
        fail("C: encode != [special id]; the single-token opener would stay "
             "unmasked — fix inert, token-id mechanism required")
        return False
    params = SamplingParams(bad_words=[OPENER])
    params.update_from_tokenizer(tokenizer)
    rows = params.bad_words_token_ids
    print(f"C: update_from_tokenizer rows = {rows}")
    if rows is None or [token_id] not in rows:
        fail("C: sampler dry run did not derive the opener mask")
        return False
    print(f"C: dry-run mask OK")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b", "c", "all"), default="all")
    parser.add_argument("--model-dir", help="serving model dir (tokenizer phase)")
    args = parser.parse_args()
    results = {}
    if args.phase in ("a", "all"):
        results["A identity"] = phase_identity()
    if args.phase in ("b", "all"):
        results["B sources"] = phase_sources()
    if args.phase in ("c", "all"):
        if not args.model_dir:
            fail("C: --model-dir required for the tokenizer phase")
            results["C tokenizer"] = False
        else:
            results["C tokenizer"] = phase_tokenizer(args.model_dir)
    for name, ok in results.items():
        print(f"== {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
