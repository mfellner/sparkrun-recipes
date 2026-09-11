#!/usr/bin/env python3
"""Host self-test and exact in-container gate for client-stop suppression."""
from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path

PRODUCTION_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py"
)
PATCH = Path(__file__).with_name("patch_suppress_stops_multitoken.py")


def synthetic_target() -> str:
    patch = runpy.run_path(str(PATCH))
    return '''class IncrementalDetokenizer:
    @staticmethod
    def _suppress_stops_enabled():
        return __import__("os").environ.get("GLM53_SUPPRESS_STOPS_IN_REASONING", "1") == "1"

    @staticmethod
    def _reasoning_stop_markers():
        return "<think>", "</think>"

    @staticmethod
    def _maybe_enable_reasoning_stop_guard(detok, tokenizer, request):
        try:
            if not IncrementalDetokenizer._suppress_stops_enabled():
                return
''' + patch["OLD"] + '''        except Exception:
            return

    def should_stop(self):
        return (
            self.stop
            and self.num_output_tokens() > self.min_tokens
''' + patch["CHECK_OLD"] + '''        )
'''


def verify_source(source: str, source_path: Path) -> None:
    assert "# [glm53-request-reasoning-stop-guard]" in source
    assert "# [glm53-stop-policy]" in source
    assert "and not IncrementalDetokenizer._suppress_stops_enabled()" not in source
    assert "and (not self._reasoning_stop_guard or self._reasoning_closed)" in source
    namespace: dict = {}
    exec(compile(source, str(source_path), "exec"), namespace)
    cls = namespace["IncrementalDetokenizer"]

    class Detok:
        stop = ["Question:"]
        _reasoning_stop_guard = False
        _reasoning_end_str = ""

    class Tokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<think>"
            return 999

    def check(state: object, prompt: list[int], expected: bool) -> None:
        request = type(
            "Request",
            (),
            {"reasoning_ended": state, "prompt_token_ids": prompt},
        )()
        detok = Detok()
        cls._maybe_enable_reasoning_stop_guard(detok, Tokenizer(), request)
        assert detok._reasoning_stop_guard is expected, (state, prompt)
        if expected:
            assert detok._reasoning_end_str == "</think>"

    check(False, [1, 2], True)
    check(True, [1, 999], False)
    check(None, [1, 999], True)
    check(None, [1, 2], False)

    old = os.environ.get("GLM53_SUPPRESS_STOPS_IN_REASONING")
    os.environ["GLM53_SUPPRESS_STOPS_IN_REASONING"] = "0"
    try:
        check(False, [1, 999], False)
    finally:
        if old is None:
            os.environ.pop("GLM53_SUPPRESS_STOPS_IN_REASONING", None)
        else:
            os.environ["GLM53_SUPPRESS_STOPS_IN_REASONING"] = old


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--production",
        action="store_true",
        help="verify the exact pinned in-container vLLM production path",
    )
    args = parser.parse_args()
    if args.production:
        if not PRODUCTION_TARGET.is_file():
            raise SystemExit(f"missing production target {PRODUCTION_TARGET}")
        verify_source(PRODUCTION_TARGET.read_text(), PRODUCTION_TARGET)
        print("request-state-bound reasoning stop policy OK (production path)")
        return 0

    patch = runpy.run_path(str(PATCH))
    patched, status = patch["apply_text"](synthetic_target())
    assert status == "repaired"
    verify_source(patched, Path("<synthetic-detokenizer>"))
    print("request-state-bound reasoning stop policy synthetic host self-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
