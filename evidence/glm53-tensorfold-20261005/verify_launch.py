#!/usr/bin/env python3
"""Validate persisted launch state, per-rank container IDs and mod bytes."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLUSTER = "sparkrun_ce1b4db30465bce3_85f608d441ab"
IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:a8067cd7e14c14fa83d1dbed60261428f6d1737cec4554445573354af040dd7c"
IMAGE_ID = "sha256:549afbcdb787fec95446c207e572acb2791488670061054cc01f526db2fe6f28"
RECIPE = ROOT / "recipes/glm-5.3-flash-exl3-tensorfold-dual-spark-1m-v171.yaml"
MOD = ROOT / "mods/glm53-tensorfold-1m"
MOD_FILES = ("SHA256SUMS", "serve.sh", "run.sh", "fabric.py")
HOSTS = ("192.168.178.47", "192.168.178.46")
JOB = "/home/max/.cache/sparkrun/jobs/ce1b4db30465bce3_85f608d441ab.yaml"
LOG = "/home/max/.hermes/cache/scratch/tf-v171-launch.log"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def probe(row: dict, argv: list[str], start: float, finish: float) -> str:
    require(row["argv"] == argv and row["returncode"] == 0 and not row["stderr"].strip(), "probe command/error")
    require(start <= row["started_at"] < row["finished_at"] <= finish, "probe time")
    return row["stdout"]


def verify(launch: dict, receipt: dict) -> dict:
    require(launch["cluster_id"] == receipt["cluster_id"] == CLUSTER, "workload ID")
    require(launch["job_path"] == JOB and launch["launch_log_path"] == LOG, "launch source")
    require(hashlib.sha256(launch["job_yaml"].encode()).hexdigest() == launch["job_sha256"], "job bytes")
    require(hashlib.sha256(launch["launch_log"].encode()).hexdigest() == launch["launch_log_sha256"], "launch log bytes")
    meta = yaml.safe_load(launch["job_yaml"])
    require(meta["cluster_id"] == CLUSTER and meta["hosts"] == list(HOSTS) and meta["recipe"] == str(RECIPE.relative_to(ROOT)), "job placement")
    require(not meta.get("api_key") and meta["sparkrun_version"] == "0.3.6", "job metadata")
    require(meta["effective_container_image"] == IMAGE and meta["recipe_state"]["_applied_overrides"] == {}, "effective launch")
    require(meta["recipe_state"]["_raw"] == yaml.safe_load(RECIPE.read_text()), "launch recipe semantics drift")
    require(launch["recipe_sha256"] == receipt["recipe_sha256"] == hashlib.sha256(RECIPE.read_bytes()).hexdigest(), "recipe bytes drift")
    require(launch["captured_at"] <= launch["completed_at"] < receipt["captured_at"], "launch state must predate acceptance")
    require(0 < launch["job_mtime_ns"] <= launch["job_ctime_ns"] <= receipt["captured_at"] * 1e9, "metadata persistence after acceptance")
    require(meta["started_at"] <= launch["job_mtime_ns"] / 1e9 + 2, "metadata timestamp")
    log = launch["launch_log"]
    for marker in ("sparkrun v0.3.6", f"Image:     {IMAGE}", "Model synced to 2 host(s)",
                   f"Cluster:   {CLUSTER}", "Head:    192.168.178.47", "Workers: 192.168.178.46"):
        require(marker in log, "launch log marker: " + marker)
    require(set(launch["hosts"]) == set(HOSTS), "launch host set")
    for rank, host in enumerate(HOSTS):
        records = launch["hosts"][host]
        require(set(records) == {"inspect", "mod_sha256"}, "rank launch records")
        name = CLUSTER + f"_node_{rank}"
        inspect = json.loads(probe(records["inspect"], ["ssh", "-o", "BatchMode=yes", host,
                                    "docker", "inspect", name], launch["captured_at"], launch["completed_at"]))
        require(len(inspect) == 1 and inspect[0]["Id"] == receipt["hosts"][host]["id"], "rank container")
        c = inspect[0]
        require(c["Image"] == IMAGE_ID and c["Config"]["Image"] == IMAGE and
                c["Config"]["Labels"]["sparkrun.cluster_id"] == CLUSTER and
                c["Config"]["Labels"]["sparkrun.rank"] == str(rank), "rank launch identity")
        created = datetime.fromisoformat(c["Created"].replace("Z", "+00:00")).timestamp()
        require(created < launch["job_mtime_ns"] / 1e9 < receipt["captured_at"], "launch-persisted recipe timing")
        require(c["Created"] == receipt["hosts"][host]["created"] and
                c["State"]["StartedAt"] == receipt["hosts"][host]["started_at"], "rank process replacement")
        cmd = ["ssh", "-o", "BatchMode=yes", host, "docker", "exec", name, "sha256sum",
               *(f"/workspace/mods/glm53-tensorfold-1m/{filename}" for filename in MOD_FILES)]
        output = probe(records["mod_sha256"], cmd, launch["captured_at"], launch["completed_at"])
        lines = output.splitlines()
        require(len(lines) == len(MOD_FILES), "mod inventory")
        for line, filename in zip(lines, MOD_FILES, strict=True):
            require(line == f"{hashlib.sha256((MOD / filename).read_bytes()).hexdigest()}  /workspace/mods/glm53-tensorfold-1m/{filename}", "mod byte drift")
    require(launch["captured_at"] <= min(row["started_at"] for records in launch["hosts"].values() for row in records.values()), "capture interval")
    return {"passed": True, "hosts": len(HOSTS), "launch_recipe_semantics": True, "mod_bytes": len(MOD_FILES)}


if __name__ == "__main__":
    print(json.dumps(verify(json.loads((HERE / "launch.json").read_text()),
                            json.loads((HERE / "receipt.json").read_text())), indent=2))
