#!/usr/bin/env python3
"""Capture raw successful-runtime evidence for the GLM-5.3 recipe."""
from __future__ import annotations
import argparse, hashlib, ipaddress, json, shlex, subprocess, time, urllib.request
from pathlib import Path

HOSTS = [
    ("192.168.178.47", "sparkrun_9000d029f34ee753_fe02508ba745_node_0"),
    ("192.168.178.46", "sparkrun_9000d029f34ee753_fe02508ba745_node_1"),
]
CLUSTER_ID = "sparkrun_9000d029f34ee753_fe02508ba745"
SINCE = "2026-09-01 16:05:00"

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
    final_acceptance_completed_at = max(
        float(json.loads((root / name).read_text())["completed_at"])
        for name in ("acceptance.json", "acceptance-xgrammar.json", "acceptance-latest-features.json")
    )
    started_at = time.time()
    captured = run(["date", "--iso-8601=seconds"])["stdout"].strip()
    rec = {
        "schema": 1,
        "captured_at": captured,
        "started_at": started_at,
        "final_acceptance_completed_at": final_acceptance_completed_at,
        "cluster_id": CLUSTER_ID,
        "kernel_window_since": SINCE,
        "capture_script_sha256": sha(Path(__file__).resolve()),
        "commands": {},
        "hosts": {},
    }
    rec["commands"]["sparkrun_status"] = run(["sparkrun", "status", "--cluster", "vacation-pair2", "--json"])
    rec["commands"]["direct_models"] = get("http://127.0.0.1:8000/v1/models")
    rec["commands"]["proxy_models"] = get("http://127.0.0.1:4000/v1/models")
    rec["commands"]["direct_marker"] = post("http://127.0.0.1:8000/v1/chat/completions", "RAW_RUNTIME_DIRECT_OK")
    rec["commands"]["proxy_marker"] = post("http://127.0.0.1:4000/v1/chat/completions", "RAW_RUNTIME_PROXY_OK")
    head_host, head_container = HOSTS[0]
    rec["commands"]["postready_ok_epoch"] = run(
        ["ssh", head_host, "docker", "exec", head_container, "stat", "-c", "%Y", "/tmp/glm53-postready.ok"]
    )
    readiness_epoch = rec["commands"]["postready_ok_epoch"]["stdout"].strip()
    if rec["commands"]["postready_ok_epoch"]["returncode"] != 0 or not readiness_epoch.isdigit():
        raise RuntimeError("failed to capture post-readiness receipt epoch")
    for host, container in HOSTS:
        h: dict[str, object] = {"container": container}
        h["docker_inspect"] = run(["ssh", host, "docker", "inspect", container])
        h["process_table"] = run(["ssh", host, "docker", "exec", container, "ps", "-eo", "pid,args"])
        h["serve_log"] = run(["ssh", host, "docker", "exec", container, "cat", "/tmp/sparkrun_serve.log"])
        h["kernel_log"] = run(["ssh", host, f"journalctl -k --since '{SINCE}' --no-pager"])
        h["kernel_log_postready"] = run(["ssh", host, f"journalctl -k --since '@{readiness_epoch}' --no-pager"])
        h["kernel_log_after_acceptance"] = run(["ssh", host, f"journalctl -k --since '@{int(final_acceptance_completed_at)}' --no-pager"])
        h["rdma_link"] = run(["ssh", host, "rdma", "link"])
        h["private_ipv4"] = private_ipv4(host)
        rate_cmd = "for d in /sys/class/infiniband/*; do h=$(basename \"$d\"); printf '%s rate=' \"$h\"; cat \"$d/ports/1/rate\"; printf '%s state=' \"$h\"; cat \"$d/ports/1/state\"; printf '%s phys_state=' \"$h\"; cat \"$d/ports/1/phys_state\"; done"
        h["hca_rate_state"] = run(["ssh", host, rate_cmd])
        overlay_cmd = (
            "sha256sum "
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py "
            "/workspace/mods/glm-5.3-flash-exl3-upstream-1m/exl3.py "
            "/cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/"
            "snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a/model.safetensors; "
            "grep -c glm53-kpool-tail-slotmap "
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/block_table.py; "
            "grep -c ABLIT-HOOK "
            "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py; "
            "sha256sum /opt/glm53/chat_template.jinja; "
            "python3 -c \"from pathlib import Path; i=Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py').read_text(); s=Path('/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py').read_text(); print('indexer_workspace_marks='+str(i.count('[glm53-indexer-workspace]'))); print('spinwait_stock='+str(s.count('busy_loop_s: float = 1,')))\"; "
            "ext=$(python3 -c \"import torch, exllamav3_ext as e; print(e.__file__)\"); "
            "sha256sum \"$ext\"; "
            "python3 -c \"import torch, exllamav3_ext as e; print('e2_symbols=' + ','.join(n for n in ('exl3_moe','exl3_fat_gemm','exl3_fat_gemm_scatter') if hasattr(e,n)))\"; "
            "env | grep -E '^(ABLIT|EXL3_|NCCL_IB_|GLM53_)' | sort"
        )
        h["overlay_runtime"] = run(
            [
                "ssh",
                host,
                f"docker exec {shlex.quote(container)} bash -lc {shlex.quote(overlay_cmd)}",
            ]
        )
        nccl_cmd = (
            "pid=$(pgrep -fo 'vllm serve'); test -n \"$pid\"; "
            "lib=$(awk '/libnccl[.]so[.]2/{print $6; exit}' /proc/$pid/maps); test -n \"$lib\"; "
            "printf 'pid=%s\\npath=%s\\n' \"$pid\" \"$(readlink -f \"$lib\")\"; "
            "printf 'sha256='; sha256sum \"$lib\" | cut -d' ' -f1; "
            "python3 -c \"import importlib.metadata as m; print('package_version='+m.version('nvidia-nccl-cu13'))\""
        )
        h["nccl_runtime"] = run(
            [
                "ssh",
                host,
                f"docker exec {shlex.quote(container)} bash -lc {shlex.quote(nccl_cmd)}",
            ]
        )
        if host == HOSTS[0][0]:
            h["postready_rc"] = run(["ssh", host, "docker", "exec", container, "cat", "/tmp/glm53-postready.rc"])
            h["postready_ok"] = run(["ssh", host, "docker", "exec", container, "cat", "/tmp/glm53-postready.ok"])
            h["postready_log"] = run(["ssh", host, "docker", "exec", container, "cat", "/tmp/glm53-postready.log"])
        rec["hosts"][host] = h
    rec["completed_at"] = time.time()
    out = root / "raw" / "runtime-success.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")

    print(json.dumps({"runtime": str(out), "cluster_id": CLUSTER_ID}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
