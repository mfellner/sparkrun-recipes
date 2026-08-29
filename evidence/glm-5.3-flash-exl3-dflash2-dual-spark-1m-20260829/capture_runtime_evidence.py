#!/usr/bin/env python3
"""Capture raw successful-runtime evidence for the GLM-5.3 recipe."""
from __future__ import annotations
import argparse, hashlib, ipaddress, json, subprocess, time, urllib.request
from pathlib import Path

HOSTS = [
    ("192.168.1.95", "sparkrun_d56463b6b176dd58_009b8bd44277_node_0"),
    ("192.168.1.111", "sparkrun_d56463b6b176dd58_009b8bd44277_node_1"),
]
CLUSTER_ID = "sparkrun_d56463b6b176dd58_009b8bd44277"
SINCE = "2026-08-29 16:55:00"

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

def private_ipv4(host: str) -> dict:
    raw = run(["ssh", host, "ip", "-j", "-4", "addr"])
    out = {"argv": raw["argv"], "returncode": raw["returncode"], "stderr": raw["stderr"], "addresses": []}
    if raw["returncode"] != 0:
        return out
    nets = [ipaddress.ip_network(x) for x in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]
    for iface in json.loads(raw["stdout"]):
        ifname = iface.get("ifname") or ""
        if ifname.startswith(("docker", "br-", "veth")):
            continue
        for addr in iface.get("addr_info", []):
            local = addr.get("local")
            if local and any(ipaddress.ip_address(local) in net for net in nets):
                out["addresses"].append({"ifname": ifname, "local": local, "prefixlen": addr.get("prefixlen")})
    return out

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
        h["private_ipv4"] = private_ipv4(host)
        rate_cmd = "for d in /sys/class/infiniband/*; do h=$(basename \"$d\"); printf '%s rate=' \"$h\"; cat \"$d/ports/1/rate\"; printf '%s state=' \"$h\"; cat \"$d/ports/1/state\"; printf '%s phys_state=' \"$h\"; cat \"$d/ports/1/phys_state\"; done"
        h["hca_rate_state"] = run(["ssh", host, rate_cmd])
        if host == HOSTS[0][0]:
            h["postready_rc"] = run(["ssh", host, "docker", "exec", container, "cat", "/tmp/glm53-postready.rc"])
            h["postready_ok"] = run(["ssh", host, "docker", "exec", container, "cat", "/tmp/glm53-postready.ok"])
            h["postready_log"] = run(["ssh", host, "docker", "exec", container, "cat", "/tmp/glm53-postready.log"])
        rec["hosts"][host] = h
    out = root / "raw" / "runtime-success.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")

    files = ["benchmark-structured.json", "benchmark-code.json", "benchmark-prose.json", "acceptance-basic-precursor.json", "acceptance-1m-features.json", "acceptance-basic-final.json", "acceptance-final-compact.json", "acceptance-xgrammar.json", "telemetry-parity.jsonl", "runtime-health.json", "raw/runtime-success.json"]
    manifest = {
        "schema": 2,
        "cluster_id": CLUSTER_ID,
        "recipe_sha256": "55526841b848320c66bb631f569e70a972f9261512b2cb4f05c79cc7286ec7f3",
        "runtime_capture": "raw/runtime-success.json",
        "processes": {
            "sparkrun_88b110d8c27f0455_73f56777180a": {
                "role": "precursor_performance_long_stress",
                "recipe_sha256": "da5c462d2322daf42449ac1996079f9af3f3bd9d1491bf819c718ac67789b1a6",
            },
            CLUSTER_ID: {
                "role": "exact_final",
                "recipe_sha256": "55526841b848320c66bb631f569e70a972f9261512b2cb4f05c79cc7286ec7f3",
            },
        },
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
            if "cluster_id" in d: item["process_id"] = d["cluster_id"]
            if "process_role" in d: item["process_role"] = d["process_role"]
        elif rel == "telemetry-parity.jsonl":
            item["process_id"] = "sparkrun_88b110d8c27f0455_73f56777180a"
            item["process_role"] = "precursor_performance_long_stress"
        manifest["artifacts"][rel] = item
    (root / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"runtime": str(out), "manifest": str(root / 'run-manifest.json')}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
