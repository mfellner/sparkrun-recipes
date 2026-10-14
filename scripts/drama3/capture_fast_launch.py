#!/usr/bin/env python3
"""Join launch-persisted SparkRun recipe with the exact running container. SPDX-License-Identifier: MIT."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import yaml
from verify_evidence import reject_secrets

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    command = ["sparkrun", "status", "--hosts", "192.168.178.70", "--json"]
    status = json.loads(subprocess.check_output(command, text=True, timeout=60))
    match = [row for row in status["solo_entries"] if row["cluster_id"] == args.workload]
    if len(match) != 1:
        raise ValueError("workload absent or duplicated in SparkRun status")
    meta = match[0]["meta"]
    persisted = Path.home() / ".cache/sparkrun/jobs" / (args.workload.removeprefix("sparkrun_") + ".yaml")
    raw = persisted.read_bytes()
    record = yaml.safe_load(raw)
    reject_secrets(record)
    recipe = args.recipe.resolve(strict=True)
    expected = yaml.safe_load(recipe.read_text())
    if record["recipe_state"]["_raw"] != expected or meta["recipe_state"]["_raw"] != expected:
        raise ValueError("launch-persisted recipe differs from reviewed YAML")
    inspect = json.loads(subprocess.check_output(["ssh", "192.168.178.70", "docker", "inspect", args.container], text=True, timeout=30))[0]
    if inspect["Config"]["Labels"]["sparkrun.cluster_id"] != args.workload or inspect["Config"]["Image"] != expected["container"]:
        raise ValueError("persisted workload is not the target container/image")
    captured = time.time()
    mtime = persisted.stat().st_mtime
    result = dict(workload_id=args.workload, container_id=inspect["Id"], image=inspect["Config"]["Image"], recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(), persistence_path=str(persisted), persistence_mtime=mtime, captured_at=captured, persisted_before_acceptance=mtime <= captured, persisted_sha256=hashlib.sha256(raw).hexdigest(), status_command=command, status_meta=meta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "job-state.yaml").write_bytes(raw)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("workload_id", "container_id", "image", "persisted_sha256")}))


if __name__ == "__main__":
    main()
