#!/usr/bin/env python3
"""Fail-closed Qwen3.8 Flash Next OpenAI API acceptance harness."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def post_json(url: str, payload: dict, timeout: int) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, time.monotonic() - start


def get_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def content_of(response: dict) -> str:
    return str(response["choices"][0]["message"].get("content") or "").strip()


def make_fixture() -> tuple[str, str]:
    image = Image.new("RGB", (720, 420))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 719, 419), fill="#f5df4d")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.ellipse((60, 120, 260, 320), fill="#1867c0")
    draw.rectangle((450, 120, 650, 320), fill="#d7191c")
    draw.text((120, 25), "VISION CODE 7K4P", fill="black", font=font)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    raw = output.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


def exact_chat(base: str, model: str, token: str, timeout: int = 600) -> dict:
    response, elapsed = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"Do not explain. Reply with exactly {token}"}],
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )
    content = content_of(response)
    if content != token:
        raise AssertionError(f"expected exact {token!r}, got {content!r}")
    return {"token": token, "content": content, "elapsed_s": elapsed, "usage": response.get("usage")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://192.168.178.48:8000")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--long-repetitions", type=int, default=40000)
    parser.add_argument("--long-timeout", type=int, default=1800)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    started = time.time()
    results: dict = {"base": base, "model": args.model, "started_unix": started}

    models = get_json(f"{base}/v1/models")
    model_rows = models.get("data", [])
    ids = [row.get("id") for row in model_rows]
    if args.model not in ids:
        raise AssertionError(f"served model {args.model!r} absent from {ids!r}")
    row = next(row for row in model_rows if row.get("id") == args.model)
    results["models"] = {"ids": ids, "max_model_len": row.get("max_model_len")}

    results["deterministic_chat"] = exact_chat(base, args.model, "QWEN38_DIRECT_OK")

    tokens = [f"Q38_C{i}_OK" for i in range(args.concurrency)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        concurrent_rows = list(pool.map(lambda token: exact_chat(base, args.model, token), tokens))
    results["concurrency"] = {"requested": args.concurrency, "passed": len(concurrent_rows), "rows": concurrent_rows}

    fixture_uri, fixture_sha = make_fixture()
    vision_response, vision_elapsed = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the code and identify both colored shapes. Include the code, both colors, and both shape names."},
                        {"type": "image_url", "image_url": {"url": fixture_uri}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        900,
    )
    vision_content = content_of(vision_response)
    vision_lower = vision_content.lower()
    for expected in ("7k4p", "blue", "circle", "red", "square"):
        if expected not in vision_lower:
            raise AssertionError(f"vision response omitted {expected!r}: {vision_content!r}")
    results["vision"] = {
        "fixture_sha256": fixture_sha,
        "content": vision_content,
        "elapsed_s": vision_elapsed,
        "usage": vision_response.get("usage"),
    }

    tool_response, tool_elapsed = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Use report_code with code q38 and count 8. Do not answer in prose."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "report_code",
                        "description": "Report a code and integer count",
                        "parameters": {
                            "type": "object",
                            "properties": {"code": {"type": "string"}, "count": {"type": "integer"}},
                            "required": ["code", "count"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        900,
    )
    calls = tool_response["choices"][0]["message"].get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"].get("name") != "report_code":
        raise AssertionError(f"unexpected tool calls: {calls!r}")
    tool_args = json.loads(calls[0]["function"]["arguments"])
    if tool_args != {"code": "q38", "count": 8}:
        raise AssertionError(f"unexpected tool args: {tool_args!r}")
    results["tool_call"] = {"name": "report_code", "arguments": tool_args, "elapsed_s": tool_elapsed}

    filler = "alpha beta gamma delta epsilon.\n"
    needle = "NEEDLE_QWEN38_LONG_CONTEXT_92MZ"
    half = args.long_repetitions // 2
    long_text = filler * half + f"\nThe unique retrieval key is {needle}.\n" + filler * (args.long_repetitions - half)
    long_response, long_elapsed = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "Retrieve the unique key from the supplied document. Reply with only that key."},
                {"role": "user", "content": long_text},
            ],
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        args.long_timeout,
    )
    long_content = content_of(long_response)
    if long_content != needle:
        raise AssertionError(f"long-context retrieval failed: {long_content!r}")
    results["long_context"] = {
        "repetitions": args.long_repetitions,
        "request_characters": len(long_text),
        "needle": needle,
        "content": long_content,
        "elapsed_s": long_elapsed,
        "usage": long_response.get("usage"),
    }

    results["finished_unix"] = time.time()
    results["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
