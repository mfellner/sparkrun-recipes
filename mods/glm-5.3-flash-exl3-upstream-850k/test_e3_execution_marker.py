#!/usr/bin/env python3
"""Deterministic contracts for the local E3 execution marker overlay."""
from __future__ import annotations

import runpy
import tempfile
from pathlib import Path

MOD = Path(__file__).resolve().parent
UPSTREAM = MOD / "upstream/overlay/exl3.py"
PATCH = MOD / "patch_e3_execution_marker.py"


def main() -> int:
    ns = runpy.run_path(str(PATCH))
    source = UPSTREAM.read_text()
    patched, status = ns["apply_text"](source)
    assert status == "patched"
    assert patched.count(ns["MARK"]) == 2
    assert patched.count(ns["CALL_NEW"]) == 1
    compile(patched, "<patched-exl3>", "exec")
    same, status = ns["apply_text"](patched)
    assert status == "skipped"
    assert same == patched

    logs: list[tuple] = []

    class Value:
        def __init__(self, value: int):
            self.value = value

        def __gt__(self, cap: int):
            return Value(sum(item > cap for item in [64, 8, 96]))

        def sum(self):
            return self

        def item(self):
            return self.value

    namespace = {
        "_EXL3_FAT_DIAG": {
            "grouped_calls": 1,
            "fat_expert_runs": 0,
            "configured_tier": "grouped",
            "effective_tier": "grouped",
        },
        "logger": type("Logger", (), {"info": lambda _self, *args: logs.append(args)})(),
    }
    exec(ns["HELPER"], namespace)
    namespace["_record_grouped_e3_execution"](Value(0), 32)
    assert namespace["_EXL3_FAT_DIAG"]["fat_expert_runs"] == 2
    assert len(logs) == 1
    rendered = logs[0][0] % logs[0][1:]
    assert rendered == (
        "[glm53-e3-executed] grouped_calls=1 fat_expert_runs=2 "
        "configured_tier=grouped effective_tier=grouped"
    )
    namespace["_EXL3_FAT_DIAG"]["grouped_calls"] = 2
    namespace["_record_grouped_e3_execution"](Value(0), 32)
    assert namespace["_EXL3_FAT_DIAG"]["fat_expert_runs"] == 2
    assert len(logs) == 1
    print("E3 execution marker overlay tests OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
