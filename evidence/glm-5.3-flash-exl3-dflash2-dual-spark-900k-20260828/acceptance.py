#!/usr/bin/env python3
"""Functional acceptance matrix for the GLM-5.3 EXL3 SparkRun recipe."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any

MODEL = "GLM-5.3-Flash-EXL3"


def post(base_url: str, path: str, body: dict[str, Any], timeout: float = 900.0) -> dict[str, Any]:
    raw = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=raw,
        headers={"Content-Type": "application/json", "User-Agent": "sparkrun-glm53-acceptance/1"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
        return {
            "http": resp.status,
            "started_at": start,
            "completed_at": time.time(),
            "request": body,
            "request_sha256": hashlib.sha256(raw).hexdigest(),
            "response": data,
        }


def get_json(base_url: str, path: str, timeout: float = 30.0) -> dict[str, Any]:
    start = time.time()
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as resp:
        return {"http": resp.status, "started_at": start, "completed_at": time.time(), "response": json.load(resp)}


def message_text(result: dict[str, Any]) -> str:
    try:
        msg = result["response"]["choices"][0]["message"]
        return (msg.get("content") or "").strip()
    except Exception:
        return ""


def exact_request(marker: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Reply with exactly {marker} and nothing else."}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def make_quadrant_png(size: int = 256) -> bytes:
    # top-left red, top-right green, bottom-left blue, bottom-right yellow
    rows = []
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            idx = (2 if y >= size // 2 else 0) + (1 if x >= size // 2 else 0)
            row.extend(colors[idx])
        rows.append(bytes(row))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + png_chunk(b"IEND", b"")


def vision_request(png: bytes) -> dict[str, Any]:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is the top-left quadrant? Reply with exactly RED."},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct-url", default="http://127.0.0.1:8000")
    ap.add_argument("--proxy-url", default="")
    ap.add_argument("--recipe", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--long-repeat", type=int, default=55000)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = make_quadrant_png()
    fixture_path = out_path.with_name("vision-quadrants.png")
    fixture_path.write_bytes(fixture)
    record: dict[str, Any] = {
        "schema": 1,
        "run_id": hashlib.sha256(f"{time.time_ns()}:{out_path}".encode()).hexdigest()[:16],
        "started_at": time.time(),
        "model": MODEL,
        "direct_url": args.direct_url,
        "proxy_url": args.proxy_url or None,
        "recipe_sha256": hashlib.sha256(Path(args.recipe).read_bytes()).hexdigest() if args.recipe else None,
        "acceptance_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture": {
            "path": fixture_path.name,
            "sha256": hashlib.sha256(fixture).hexdigest(),
            "size": len(fixture),
            "description": "256x256 RGB PNG: red TL, green TR, blue BL, yellow BR",
        },
        "checks": {},
    }

    models = get_json(args.direct_url, "/v1/models")
    model_ok = any(m.get("id") == MODEL for m in models["response"].get("data", []))
    record["checks"]["direct_models"] = {"passed": model_ok, **models}

    direct = post(args.direct_url, "/v1/chat/completions", exact_request("GLM53_DIRECT_OK"))
    record["checks"]["direct_exact"] = {"passed": message_text(direct) == "GLM53_DIRECT_OK", **direct}

    markers = [f"GLM53_C4_{i}_OK" for i in range(4)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(post, args.direct_url, "/v1/chat/completions", exact_request(marker)) for marker in markers]
        concurrent_rows = [f.result() for f in futures]
    record["checks"]["direct_concurrency_c4"] = {
        "passed": all(message_text(row) == marker for row, marker in zip(concurrent_rows, markers)),
        "markers": markers,
        "results": concurrent_rows,
    }

    needle = "NEEDLE_GL53_842917"
    filler = "alpha " * args.long_repeat
    long_prompt = filler + f"\nHidden retrieval code: {needle}\n" + filler + f"\nReply with exactly {needle}."
    long_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": long_prompt}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    long_result = post(args.direct_url, "/v1/chat/completions", long_body, timeout=1800)
    record["checks"]["direct_long_context"] = {
        "passed": message_text(long_result) == needle,
        "needle": needle,
        "filler_token": "alpha ",
        "filler_repeats_each_side": args.long_repeat,
        "prompt_chars": len(long_prompt),
        "prompt_sha256": hashlib.sha256(long_prompt.encode()).hexdigest(),
        **long_result,
    }

    tool_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Use the weather tool for Paris, France."}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }
    tool_result = post(args.direct_url, "/v1/chat/completions", tool_body)
    try:
        calls = tool_result["response"]["choices"][0]["message"].get("tool_calls") or []
        tool_ok = any(call.get("function", {}).get("name") == "get_weather" for call in calls)
    except Exception:
        tool_ok = False
    record["checks"]["direct_tool_call"] = {"passed": tool_ok, **tool_result}

    vision = post(args.direct_url, "/v1/chat/completions", vision_request(fixture), timeout=900)
    record["checks"]["direct_vision"] = {"passed": message_text(vision) == "RED", **vision}

    if args.proxy_url:
        proxy_models = get_json(args.proxy_url, "/v1/models")
        proxy_model_ok = any(m.get("id") == MODEL for m in proxy_models["response"].get("data", []))
        record["checks"]["proxy_models"] = {"passed": proxy_model_ok, **proxy_models}
        proxy = post(args.proxy_url, "/v1/chat/completions", exact_request("GLM53_PROXY_OK"))
        record["checks"]["proxy_exact"] = {"passed": message_text(proxy) == "GLM53_PROXY_OK", **proxy}
        proxy_vision = post(args.proxy_url, "/v1/chat/completions", vision_request(fixture), timeout=900)
        record["checks"]["proxy_vision"] = {"passed": message_text(proxy_vision) == "RED", **proxy_vision}

    record["completed_at"] = time.time()
    record["passed"] = all(check.get("passed") is True for check in record["checks"].values())
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v["passed"] for k, v in record["checks"].items()}, indent=2))
    print(f"wrote {out_path}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
