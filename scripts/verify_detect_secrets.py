#!/usr/bin/env python3
"""Verify the reviewed detect-secrets baseline against all files."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def rows(document: dict) -> list[dict]:
    return [row for findings in document.get("results", {}).values() for row in findings]


def identities(document: dict) -> Counter[tuple[str, str, str]]:
    return Counter(
        (row["filename"], row["type"], row["hashed_secret"])
        for row in rows(document)
    )


def scan_process_errors(returncode: int, stdout: str, stderr: str) -> list[str]:
    errors = []
    if returncode != 0:
        errors.append(f"detect-secrets exited {returncode}")
    if stdout:
        errors.append("baseline update unexpectedly wrote stdout")
    if stderr:
        errors.append("detect-secrets wrote stderr")
    return errors


def raw_occurrence_summary(root: Path, document: dict) -> dict:
    executable = shutil.which("detect-secrets")
    if executable is None:
        raise RuntimeError("detect-secrets executable not found")
    shebang = Path(executable).read_text().splitlines()[0]
    if not shebang.startswith("#!"):
        raise RuntimeError("detect-secrets executable has no Python shebang")
    python = shebang[2:].strip().split()[0]
    program = r'''import hashlib, json, sys
from collections import Counter
from pathlib import Path
from detect_secrets.core.scan import scan_file
from detect_secrets.settings import transient_settings
root = Path(sys.argv[1])
document = json.load(sys.stdin)
occurrences = Counter()
with transient_settings(document):
    for filename in sorted(document.get("results", {})):
        for finding in scan_file(str(root / filename)):
            occurrences[(filename, finding.type, finding.secret_hash)] += 1
payload = json.dumps(
    [(*identity, count) for identity, count in sorted(occurrences.items())],
    separators=(",", ":"),
).encode()
print(json.dumps({
    "identities": len(occurrences),
    "occurrences": sum(occurrences.values()),
    "sha256": hashlib.sha256(payload).hexdigest(),
}, sort_keys=True))
'''
    process = subprocess.run(
        [python, "-c", program, str(root)],
        input=json.dumps(document),
        text=True,
        capture_output=True,
        timeout=180,
    )
    if process.returncode != 0 or process.stderr:
        raise RuntimeError("raw detect-secrets occurrence scan failed")
    return json.loads(process.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    baseline_path = root / ".secrets.baseline"
    report = {
        "passed": False,
        "findings": 0,
        "adjudicated_false_positives": 0,
        "stderr": "",
        "raw_occurrences": 0,
        "errors": [],
    }

    try:
        reviewed = json.loads(baseline_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        report["errors"].append(f"baseline unreadable: {type(exc).__name__}")
        print(json.dumps(report, sort_keys=True))
        return 1

    reviewed_rows = rows(reviewed)
    report["findings"] = len(reviewed_rows)
    report["adjudicated_false_positives"] = sum(
        row.get("is_secret") is False for row in reviewed_rows
    )
    if not reviewed_rows:
        report["errors"].append("baseline contains no reviewed findings")
    if report["adjudicated_false_positives"] != report["findings"]:
        report["errors"].append("baseline has unadjudicated or secret findings")

    try:
        expected_occurrences = json.loads(
            (root / ".secrets.occurrences.json").read_text()
        )
        current_occurrences = raw_occurrence_summary(root, reviewed)
        report["raw_occurrences"] = current_occurrences["occurrences"]
        if current_occurrences != expected_occurrences:
            report["errors"].append(
                "raw finding occurrences differ from reviewed manifest"
            )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        report["errors"].append(
            f"occurrence manifest verification failed: {type(exc).__name__}"
        )

    with tempfile.TemporaryDirectory(prefix="detect-secrets-review-") as tmp:
        candidate = Path(tmp) / ".secrets.baseline"
        shutil.copy2(baseline_path, candidate)
        process = subprocess.run(
            [
                "detect-secrets", "scan", "--all-files", "--no-verify",
                "--baseline", str(candidate), ".",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=180,
        )
        report["stderr"] = process.stderr
        report["errors"].extend(scan_process_errors(
            process.returncode,
            process.stdout,
            process.stderr,
        ))
        try:
            rescanned = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            report["errors"].append(f"rescanned baseline unreadable: {type(exc).__name__}")
            rescanned = {"results": {}}

    if identities(rescanned) != identities(reviewed):
        report["errors"].append("current findings differ from reviewed baseline")
    if any(row.get("is_secret") is not False for row in rows(rescanned)):
        report["errors"].append("rescan contains unadjudicated findings")

    report["passed"] = not report["errors"]
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
