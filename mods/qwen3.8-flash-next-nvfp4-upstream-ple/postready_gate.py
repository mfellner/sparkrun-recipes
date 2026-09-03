#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn

serve_pid = int(sys.argv[1])
port = int(os.environ.get("QWEN38_SERVE_PORT", "8000"))
host = os.environ.get("QWEN38_SERVE_HOST", "127.0.0.1")
model = os.environ.get("QWEN38_SERVED_MODEL", "qwen3.8-flash-next")
timeout_s = int(os.environ.get("QWEN38_ENGINE_READY_TIMEOUT_S", "3600"))
base = f"http://{host}:{port}"
receipt = Path("/tmp/qwen38-postready.json")
started = time.time()

def write_receipt(data: dict) -> None:
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)

def alive() -> bool:
    try:
        os.kill(serve_pid, 0)
        return True
    except ProcessLookupError:
        return False

def fail(message: str, details=None) -> NoReturn:
    data = {
        "ok": False,
        "message": message,
        "model": model,
        "port": port,
        "started_unix": started,
        "finished_unix": time.time(),
        "details": details,
    }
    if alive():
        os.kill(serve_pid, signal.SIGTERM)
    try:
        write_receipt(data)
    except OSError as exc:
        print(f"[QWEN38 GATE] receipt write failed: {exc!r}", file=sys.stderr, flush=True)
    print(f"[QWEN38 GATE] FATAL: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def handle_unexpected(exc_type, exc_value, exc_traceback):
    if alive():
        os.kill(serve_pid, signal.SIGTERM)
    data = {
        "ok": False,
        "message": "unhandled post-readiness gate exception",
        "model": model,
        "port": port,
        "started_unix": started,
        "finished_unix": time.time(),
        "details": {"type": exc_type.__name__, "error": repr(exc_value)},
    }
    try:
        write_receipt(data)
    except OSError:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_unexpected

try:
    receipt.unlink(missing_ok=True)
except OSError as exc:
    fail("could not remove stale post-readiness receipt", repr(exc))

def get_json(url: str, timeout: int):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def post_json(url: str, payload: dict, timeout: int):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

deadline = time.monotonic() + timeout_s
models = None
while time.monotonic() < deadline:
    if not alive():
        fail("vLLM exited before API readiness")
    try:
        candidate = get_json(f"{base}/v1/models", 10)
        if any(item.get("id") == model for item in candidate.get("data", [])):
            models = candidate
            break
    except (OSError, ValueError, urllib.error.URLError):
        pass
    time.sleep(5)

if models is None:
    fail(f"/v1/models did not advertise {model!r} within {timeout_s}s")

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly QWEN38_OK and nothing else.",
        }
    ],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": False},
}
try:
    completion = post_json(f"{base}/v1/chat/completions", payload, 900)
except Exception as exc:
    fail("semantic completion request failed", repr(exc))

try:
    choice = completion["choices"][0]
    content = choice["message"].get("content", "").strip()
    finish_reason = choice.get("finish_reason")
except Exception as exc:
    fail("semantic completion response shape was invalid", {"error": repr(exc), "response": completion})

if content != "QWEN38_OK" or finish_reason != "stop":
    fail(
        "semantic completion did not return the exact required answer",
        {"content": content, "finish_reason": finish_reason, "response": completion},
    )

result = {
    "ok": True,
    "model": model,
    "port": port,
    "started_unix": started,
    "finished_unix": time.time(),
    "models": models,
    "request": payload,
    "completion": completion,
}
write_receipt(result)
print("[QWEN38 GATE] PASS: model discovery and exact semantic completion", flush=True)
