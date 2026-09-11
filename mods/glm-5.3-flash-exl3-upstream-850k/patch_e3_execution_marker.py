#!/usr/bin/env python3
"""Add a one-time marker after the first positive grouped E3 execution."""
from __future__ import annotations

import os
from pathlib import Path

MARK = "# [glm53-e3-execution-marker]"
GLOBAL_ANCHOR = "_exl3_fat_tier_logged = False\n"
HELPER = f'''{MARK}
_exl3_e3_execution_logged = False


def _record_grouped_e3_execution(counts, cap):
    global _exl3_e3_execution_logged
    if _exl3_e3_execution_logged:
        return
    fat_experts = int((counts > cap).sum().item())
    if fat_experts <= 0:
        return
    _EXL3_FAT_DIAG["fat_expert_runs"] += fat_experts
    diag = _EXL3_FAT_DIAG
    logger.info(
        "[glm53-e3-executed] grouped_calls=%d fat_expert_runs=%d "
        "configured_tier=%s effective_tier=%s",
        diag["grouped_calls"],
        diag["fat_expert_runs"],
        diag["configured_tier"],
        diag["effective_tier"],
    )
    _exl3_e3_execution_logged = True
'''
CALL_OLD = '''        apply_exl3_grouped_fat(
            xh, out, counts, token_sorted, weight_sorted, layer, cap, limit
        )
        _record_exl3_fat_tier(layer, "grouped", "grouped_ok")
'''
CALL_NEW = '''        apply_exl3_grouped_fat(
            xh, out, counts, token_sorted, weight_sorted, layer, cap, limit
        )
        _record_grouped_e3_execution(counts, cap)  # [glm53-e3-execution-marker]
        _record_exl3_fat_tier(layer, "grouped", "grouped_ok")
'''


def apply_text(text: str) -> tuple[str, str]:
    marker_count = text.count(MARK)
    if marker_count == 2:
        if HELPER not in text or text.count(CALL_NEW) != 1:
            raise ValueError("partial or altered E3 execution marker state")
        return text, "skipped"
    if marker_count:
        raise ValueError(f"partial E3 execution marker state: markers={marker_count}")
    if text.count(GLOBAL_ANCHOR) != 1:
        raise ValueError("expected one E3 marker global anchor")
    if text.count(CALL_OLD) != 1:
        raise ValueError("expected one grouped E3 call anchor")
    text = text.replace(GLOBAL_ANCHOR, GLOBAL_ANCHOR + "\n" + HELPER + "\n", 1)
    text = text.replace(CALL_OLD, CALL_NEW, 1)
    compile(text, "exl3.py", "exec")
    return text, "patched"


def main() -> int:
    target = Path(os.environ.get("GLM53_OPT", "/opt/glm53")) / "exl3.py"
    if not target.is_file():
        raise SystemExit(f"missing {target}")
    try:
        patched, status = apply_text(target.read_text())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if status == "patched":
        target.write_text(patched)
    print(f"{target}: E3 execution marker {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
