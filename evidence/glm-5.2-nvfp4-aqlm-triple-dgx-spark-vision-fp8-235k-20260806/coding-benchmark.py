#!/usr/bin/env python3
"""Reproduce the three-run TypeScript coding decode measurement."""

import argparse
import json
import statistics
import time
from pathlib import Path

from benchmark import one

PROMPT = (
    "Implement a production-quality TypeScript HTTP service with configuration "
    "loading, structured logging, request validation, error handling, graceful "
    "shutdown, and tests. Return code with concise explanations."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.0.184:8000/v1")
    parser.add_argument("--output", type=Path, default=Path("coding-results.json"))
    args = parser.parse_args()

    rows = []
    for _ in range(3):
        row = one(args.base_url, PROMPT, 512)
        rows.append(row)
        print(json.dumps(row), flush=True)
        time.sleep(1)

    result = {
        "runs": rows,
        "median_decode_tps": statistics.median(row["decode_tps"] for row in rows),
        "median_ttft_ms": statistics.median(row["ttft_ms"] for row in rows),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
