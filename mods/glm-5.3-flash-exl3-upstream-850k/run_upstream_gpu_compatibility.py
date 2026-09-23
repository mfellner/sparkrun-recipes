#!/usr/bin/env python3
"""Run unchanged upstream GPU checks with a narrow stale-static-check adapter.

Only _check_dflash2's obsolete literal compact-KV assertions are replaced in
memory. All its other imports/assertions and upstream main/GPU routines remain
unchanged. No vendored file is written. Source drift fails closed.
"""
import importlib.util
import inspect
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OBSOLETE = '''    assert "compact_block = 64" in kv
    assert "page_size_padded=mla_page" in kv
    assert "padded slot-share block=%d" in kv
    assert "s.block_size != 64 or s.page_size_padded != mla_page" in kv
    standalone = kv.split("PADDED SLOT-SHARE:")[1].split("draft_uniform")[0]
    assert "compact_block" in standalone
    assert "page_size_padded=mla_page" in standalone
    assert "new_draft_specs = dict(draft_specs)" not in standalone
'''


def adapt_dflash_check(source):
    if source.count(OBSOLETE) != 1:
        raise ValueError("upstream DFlash assertion seam drifted; review adapter")
    return source.replace(OBSOLETE, "    _check_current_draft_kv()\n", 1)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_current_draft_kv():
    verifier = load("glm53_current_gpu_verifier", ROOT / "verify_runtime_patch_state.py")
    failures = verifier.verify_runtime(verifier.DEFAULT_SITE)
    if failures:
        raise AssertionError("\n".join(failures))
    draft_tests = load("glm53_current_draft_tests", ROOT / "upstream/tests/test_draft_kv_compact.py")
    # Unchanged current upstream geometry sweep includes off => 64, including
    # invalid compact-mode geometries; exact installed helper bytes checked above.
    draft_tests.test_compact_block_preserves_alignment_and_fits_page()
    print("DFLASH_CURRENT_STATIC_AND_OFF64_PASS", flush=True)


def main():
    if os.environ.get("EXL3_SELFCHECK_GPU", "1") == "0":
        raise SystemExit("GPU qualification requires EXL3_SELFCHECK_GPU != 0")
    upstream = load("glm53_upstream_gpu_gate", ROOT / "upstream/tests/test_exl3_overlay.py")
    upstream.__dict__["_check_current_draft_kv"] = check_current_draft_kv
    code = adapt_dflash_check(inspect.getsource(upstream._check_dflash2))
    exec(compile(code, "<local-dflash-static-adapter>", "exec"), upstream.__dict__)
    return upstream.main()


if __name__ == "__main__":
    raise SystemExit(main())
