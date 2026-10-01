#!/usr/bin/env python3
"""Bounded, semantic live acceptance for the pinned two-rank TensorFold run.

Run on the head after /v1/models is ready. Writes raw request/response pairs
and process identity to receipt.json, then rejects HTTP or semantic failures.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RECIPE = ROOT / "recipes/glm-5.3-flash-exl3-tensorfold-dual-spark-1m.yaml"
MOD = ROOT / "mods/glm53-tensorfold-1m"
MODEL = "GLM-5.3-Flash-EXL3"
DIRECT = "http://192.168.178.47:8000"
PROXY = "http://192.168.178.47:4000"
HOSTS = ("192.168.178.47", "192.168.178.46")
CLUSTER = "sparkrun_98fa271a0dea8179_9e378c82c913"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get(url: str, timeout: int = 20) -> dict:
    started = time.time()
    with urllib.request.urlopen(url, timeout=timeout) as response:
        result = {"url": url, "http": response.status, "body": json.loads(response.read())}
    return {**result, "started_at": started, "finished_at": time.time()}


def chat(base: str, payload: dict, timeout: int = 240) -> dict:
    started = time.time()
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    finished = time.time()
    try:
        parsed = json.loads(body)
    except (UnicodeError, json.JSONDecodeError):
        parsed = {"raw_body_b64": base64.b64encode(body).decode()}
    return {"url": base + "/v1/chat/completions", "started_at": started, "finished_at": finished,
            "http": status, "request": payload, "response": parsed}


def png_fixture() -> bytes:
    w, h = 64, 32
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * (w // 2) + b"\x00\x00\xff" * (w // 2) for _ in range(h))
    def chunk(tag: bytes, content: bytes) -> bytes:
        return struct.pack("!I", len(content)) + tag + content + struct.pack("!I", zlib.crc32(tag + content))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def plain(text: str, max_tokens: int = 80) -> dict:
    return {"model": MODEL, "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}


def record() -> dict:
    status = json.loads(subprocess.check_output(["sparkrun", "status", "--json"], text=True))
    assert CLUSTER in status["groups"], "expected SparkRun workload is absent"
    group = status["groups"][CLUSTER]
    records = {"cluster_id": CLUSTER, "recipe_sha256": digest(RECIPE), "mod_manifest_sha256": digest(MOD / "SHA256SUMS"),
               "captured_at": time.time(), "hosts": {}, "models": {}, "health_before": {}, "tests": {}}
    for rank, host in enumerate(HOSTS):
        name = CLUSTER + f"_node_{rank}"
        raw = subprocess.check_output(["ssh", host, "docker", "inspect", name], text=True)
        container = json.loads(raw)[0]
        assert container["State"]["Running"] and not container["State"]["OOMKilled"]
        cfg = container["Config"]
        hc = container["HostConfig"]
        records["hosts"][host] = {
            "rank": rank, "container": name, "id": container["Id"], "created": container["Created"],
            "started_at": container["State"]["StartedAt"], "image_id": container["Image"],
            "repo_digests": json.loads(subprocess.check_output(["ssh", host, "docker", "image", "inspect", cfg["Image"]], text=True))[0]["RepoDigests"],
            "runtime": cfg.get("Labels", {}).get("sparkrun.runtime"), "user": cfg.get("User"),
            "network_mode": hc.get("NetworkMode"), "ipc_mode": hc.get("IpcMode"),
            "privileged": hc.get("Privileged"), "cap_add": hc.get("CapAdd"),
            "no_new_privileges": hc.get("SecurityOpt"), "devices": hc.get("Devices"),
        }
    records["models"]["direct"] = get(DIRECT + "/v1/models")
    records["models"]["proxy"] = get(PROXY + "/v1/models")
    records["health_before"] = get(DIRECT + "/health")
    records["tests"]["direct"] = chat(DIRECT, plain("Reply with exactly DIRECT_OK."))
    records["tests"]["proxy"] = chat(PROXY, plain("Reply with exactly PROXY_OK."))
    prompts = [plain(f"Reply with exactly WAVE_{i}.") for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records["tests"]["concurrent"] = list(pool.map(lambda p: chat(DIRECT, p), prompts))
    fixture = png_fixture()
    records["fixture_png_sha256"] = hashlib.sha256(fixture).hexdigest()
    image_payload = {"model": MODEL, "temperature": 0, "max_tokens": 128,
                     "chat_template_kwargs": {"enable_thinking": False},
                     "messages": [{"role": "user", "content": [
                         {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(fixture).decode()}},
                         {"type": "text", "text": "What color fills the LEFT HALF of the image? Answer one color word."}]}]}
    records["tests"]["vision"] = chat(DIRECT, image_payload, 240)
    tools = [{
        "type": "function",
        "function": {
            "name": "add_tags",
            "description": "Attach tags to a document.",
            "parameters": {
                "type": "object", "required": ["doc_id", "tags"],
                "properties": {"doc_id": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}},
            },
        },
    }]
    records["tests"]["tool"] = chat(DIRECT, {"model": MODEL, "temperature": 0, "max_tokens": 1024, "tools": tools,
        "messages": [{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."}]}, 600)
    needle = "SABLE-ORCHID-7294"
    body = "\n".join(f"Archive row {i}: blue quartz transit ledger 1847 stable."
                     + (f" The only verification code is {needle}." if i == 1837 else "") for i in range(3300))
    long_payload = plain("Find the only verification code in this archive. Reply with the exact code only.\n" + body, 96)
    tokenize_payload = {"model": MODEL, "messages": long_payload["messages"]}
    tokenize_url = DIRECT + "/tokenize"
    tokenize = urllib.request.Request(tokenize_url, json.dumps(tokenize_payload).encode(), {"Content-Type": "application/json"})
    tokenize_started = time.time()
    try:
        with urllib.request.urlopen(tokenize, timeout=180) as response:
            records["long_tokenize"] = {"url": tokenize_url, "request": tokenize_payload, "http": response.status, "body": json.loads(response.read())}
    except urllib.error.HTTPError as error:
        records["long_tokenize"] = {"url": tokenize_url, "request": tokenize_payload, "http": error.code, "body": json.loads(error.read())}
    records["long_tokenize"].update(started_at=tokenize_started, finished_at=time.time())
    records["tests"]["long_needle"] = chat(DIRECT, long_payload, 600)
    records["health_after"] = get(DIRECT + "/health")
    records["completed_at"] = time.time()
    return records


if __name__ == "__main__":
    result = record()
    out = HERE / "receipt.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {out} ({out.stat().st_size} bytes); verify separately with verify.py")
