#!/usr/bin/env python3
"""Bind client-stop suppression to the pinned request reasoning state."""
from __future__ import annotations

import ast
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py")
MARK = "# [glm53-request-reasoning-stop-guard]"
CHECK_MARK = "# [glm53-stop-policy]"
OLD = '''            stop = getattr(detok, "stop", None)
            ptids = getattr(request, "prompt_token_ids", None)
            if not stop or not ptids:
                return
            start_str, end_str = IncrementalDetokenizer._reasoning_stop_markers()
            think_id = None
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert):
                think_id = convert(start_str)
            if think_id is None or think_id < 0:
                encode = getattr(tokenizer, "encode", None)
                if callable(encode):
                    try:
                        ids = encode(start_str, add_special_tokens=False)
                    except TypeError:
                        ids = encode(start_str)
                    if isinstance(ids, list) and len(ids) == 1:
                        think_id = ids[0]
            if think_id is not None and think_id >= 0 and ptids[-1] == think_id:
                detok._reasoning_stop_guard = True
                detok._reasoning_end_str = end_str
'''
BAD_NEW = '''            stop = getattr(detok, "stop", None)
            ptids = getattr(request, "prompt_token_ids", None)
            if not stop or not ptids:
                return
            start_str, end_str = IncrementalDetokenizer._reasoning_stop_markers()
            # [glm53-default-thinking-stop-guard] This published profile is
            # default-thinking. Request-level reasoning state is not preserved
            # reliably by the pinned engine, so any supplied client stop remains
            # dormant until </think>. EOS and max_tokens remain unchanged. Set
            # GLM53_SUPPRESS_STOPS_IN_REASONING=0 to restore stock stop behavior.
            detok._reasoning_stop_guard = True
            detok._reasoning_end_str = end_str
'''
NEW = '''            stop = getattr(detok, "stop", None)
            if not stop:
                return
            reasoning_ended = getattr(request, "reasoning_ended", None)
            if reasoning_ended is True:
                return
            start_str, end_str = IncrementalDetokenizer._reasoning_stop_markers()
            should_arm = reasoning_ended is False
            ptids = getattr(request, "prompt_token_ids", None)
            if reasoning_ended is None and ptids:
                # [glm53-request-reasoning-stop-guard] Older compatible paths do
                # not carry reasoning_ended; retain upstream's prompt suffix gate.
                think_id = None
                convert = getattr(tokenizer, "convert_tokens_to_ids", None)
                if callable(convert):
                    think_id = convert(start_str)
                if think_id is None or think_id < 0:
                    encode = getattr(tokenizer, "encode", None)
                    if callable(encode):
                        try:
                            ids = encode(start_str, add_special_tokens=False)
                        except TypeError:
                            ids = encode(start_str)
                        if isinstance(ids, list) and len(ids) == 1:
                            think_id = ids[0]
                should_arm = (
                    think_id is not None
                    and think_id >= 0
                    and ptids[-1] == think_id
                )
            if should_arm:
                detok._reasoning_stop_guard = True
                detok._reasoning_end_str = end_str
'''
CHECK_OLD = '''            and (not self._reasoning_stop_guard or self._reasoning_closed)
'''
CHECK_BAD = '''            # [glm53-stop-policy] This default-thinking profile ignores
            # client stop strings while the recipe guard is enabled. EOS and
            # max_tokens remain active; env opt-out restores stock checks.
            and not IncrementalDetokenizer._suppress_stops_enabled()
'''
CHECK_NEW = '''            # [glm53-stop-policy] Honor stops for non-thinking requests and
            # resume matching immediately after the reasoning end marker.
            and (not self._reasoning_stop_guard or self._reasoning_closed)
'''


def apply_text(source: str) -> tuple[str, str]:
    """Apply the state-bound policy or repair the prior global suppression."""
    text = source
    changed = False
    if NEW not in text:
        if text.count(BAD_NEW) == 1:
            text = text.replace(BAD_NEW, NEW, 1)
            changed = True
        elif text.count(OLD) == 1:
            text = text.replace(OLD, NEW, 1)
            changed = True
        else:
            raise ValueError(
                f"expected one reasoning-arm anchor, found old={text.count(OLD)} "
                f"bad={text.count(BAD_NEW)}"
            )
    if CHECK_NEW not in text:
        if text.count(CHECK_BAD) == 1:
            text = text.replace(CHECK_BAD, CHECK_NEW, 1)
            changed = True
        elif text.count(CHECK_OLD) == 1:
            text = text.replace(CHECK_OLD, CHECK_NEW, 1)
            changed = True
        else:
            raise ValueError(
                f"expected one stop-check anchor, found old={text.count(CHECK_OLD)} "
                f"bad={text.count(CHECK_BAD)}"
            )
    if text.count(NEW) != 1 or text.count(CHECK_NEW) != 1:
        raise ValueError("reasoning stop policy is duplicated or incomplete")
    if BAD_NEW in text or CHECK_BAD in text:
        raise ValueError("prior global stop suppression remains")
    ast.parse(text, filename=str(P))
    return text, "repaired" if changed else "skipped"


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    source = P.read_text()
    try:
        patched, status = apply_text(source)
    except ValueError as exc:
        raise SystemExit(f"{P}: {exc}") from exc
    if patched != source:
        P.write_text(patched)
    print(f"{status} request-state-bound reasoning stop policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
