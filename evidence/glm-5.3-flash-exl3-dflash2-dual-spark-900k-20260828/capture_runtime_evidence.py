#!/usr/bin/env python3
"""Capture raw successful-runtime evidence for the GLM-5.3 recipe."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, time, urllib.request
from pathlib import Path

HOSTS = [
    ("192.168.1.95", "sparkrun_73f38a238771a1ea_f80511f7c8ce_node_0"),
    ("192.168.1.111", "sparkrun_73f38a238771a1ea_f80511f7c8ce_node_1"),
]
CLUSTER_ID = "sparkrun_73f38a238771a1ea_f80511f7c8ce"
SINCE = "2026-08-28 18:16:00"

def run(argv: list[str], timeout: int = 180) -> dict:
    p = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

def get(url: str) -> dict:
    started = time.time()
    with urllib.request.urlopen(url, timeout=30) as r:
        body = r.read().decode()
        return {"url": url, "started_at": started, "completed_at": time.time(), "http": r.status, "body": body}

def post(url: str, marker: str) -> dict:
    body = json.dumps({"model":"GLM-5.3-Flash-EXL3","messages":[{"role":"user","content":f"Reply with exactly {marker}"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read().decode()
        return {"url": url, "marker": marker, "request_sha256": hashlib.sha256(body).hexdigest(), "started_at": started, "completed_at": time.time(), "http": r.status, "body": raw}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", required=True)
    args = ap.parse_args()
    root = Path(args.evidence_dir).resolve()
    captured = run(["date", "--iso-8601=seconds"])["stdout"].strip()
    rec = {
        "schema": 1,
        "captured_at": captured,
        "cluster_id": CLUSTER_ID,
        "kernel_window_since": SINCE,
        "capture_script_sha256": sha(Path(__file__).resolve()),
        "commands": {},
        "hosts": {},
    }
    rec["commands"]["sparkrun_status"] = run(["sparkrun", "status", "--cluster", "vacation-pair2", "--json"])
    rec["commands"]["direct_models"] = get("http://127.0.0.1:8000/v1/models")
    rec["commands"]["proxy_models"] = get("http://192.168.1.95:4000/v1/models")
    rec["commands"]["direct_marker"] = post("http://127.0.0.1:8000/v1/chat/completions", "RAW_RUNTIME_DIRECT_OK")
    rec["commands"]["proxy_marker"] = post("http://192.168.1.95:4000/v1/chat/completions", "RAW_RUNTIME_PROXY_OK")
    for host, container in HOSTS:
        h: dict[str, object] = {"container": container}
        h["docker_inspect"] = run(["ssh", host, "docker", "inspect", container])
        h["serve_log"] = run(["ssh", host, "docker", "exec", container, "cat", "/tmp/sparkrun_serve.log"])
        h["kernel_log"] = run(["ssh", host, f"journalctl -k --since '{SINCE}' --no-pager"])
        h["rdma_link"] = run(["ssh", host, "rdma", "link"])
        h["ip_addr"] = run(["ssh", host, "ip", "-br", "addr"])
        rate_cmd = "for d in /sys/class/infiniband/*; do h=$(basename \"$d\"); printf '%s rate=' \"$h\"; cat \"$d/ports/1/rate\"; printf '%s state=' \"$h\"; cat \"$d/ports/1/state\"; printf '%s phys_state=' \"$h\"; cat \"$d/ports/1/phys_state\"; done"
        h["hca_rate_state"] = run(["ssh", host, rate_cmd])
        rec["hosts"][host] = h
    out = root / "raw" / "runtime-success.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")

    files = ["benchmark-structured.json", "benchmark-code.json", "benchmark-prose.json", "acceptance.json", "telemetry-parity.jsonl", "runtime-health.json", "raw/runtime-success.json"]
    manifest = {
        "schema": 1,
        "cluster_id": CLUSTER_ID,
        "recipe_sha256": "add8d048335618b0d0f9be221bb7038e7e968c2c6f5ce0606e0c4d3332070aff",
        "runtime_capture": "raw/runtime-success.json",
        "artifacts": {},
    }
    for rel in files:
        p = root / rel
        item = {"sha256": sha(p)}
        if p.suffix == ".json":
            d = json.loads(p.read_text())
            if "run_id" in d: item["run_id"] = d["run_id"]
            if "started_at" in d: item["started_at"] = d["started_at"]
            if "completed_at" in d: item["completed_at"] = d["completed_at"]
            if "cluster_id" in d: item["cluster_id"] = d["cluster_id"]
        manifest["artifacts"][rel] = item
    (root / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"runtime": str(out), "manifest": str(root / 'run-manifest.json')}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
