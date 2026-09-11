#!/usr/bin/env python3
"""Functional acceptance matrix for the GLM-5.3 EXL3 SparkRun recipe."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

MODEL = "GLM-5.3-Flash-EXL3"
REASONING_STOP_REASON = 154827
REMOTE_MEDIA_ERROR = {
    "error": {
        "message": "The URL must be from one of the allowed domains: ['media.invalid']. Input URL domain: 127.0.0.1",
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    }
}
PROXY_REMOTE_MEDIA_ERROR = {
    "error": {
        "code": "400",
        "message": (
            "litellm.BadRequestError: OpenAIException - The URL must be from one of the allowed domains: "
            "['media.invalid']. Input URL domain: 127.0.0.1. Received Model Group=GLM-5.3-Flash-EXL3\n"
            "Available Model Group Fallbacks=None"
        ),
        "param": None,
        "type": None,
    }
}
VIDEO_LIMIT_MESSAGE = (
    "At most 0 video(s) may be provided in one prompt. "
    "Set `--limit-mm-per-prompt` to increase this limit. (parameter=video)"
)
VIDEO_LIMIT_ERROR = {
    "error": {
        "message": VIDEO_LIMIT_MESSAGE,
        "type": "BadRequestError",
        "param": "video",
        "code": 400,
    }
}
PROXY_VIDEO_LIMIT_ERROR = {
    "error": {
        "code": "400",
        "message": (
            "litellm.BadRequestError: OpenAIException - " + VIDEO_LIMIT_MESSAGE
            + ". Received Model Group=GLM-5.3-Flash-EXL3\n"
            "Available Model Group Fallbacks=None"
        ),
        "param": None,
        "type": None,
    }
}


def write_signal(path: Path, state: str, **fields: Any) -> None:
    path.write_text(json.dumps({"state": state, **fields}) + "\n")


def wait_for_collector(path: Path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if json.loads(path.read_text()).get("state") == "collector_ready":
                return
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.1)
    raise TimeoutError("telemetry collector did not become ready")


def post(base_url: str, path: str, body: dict[str, Any], timeout: float = 900.0, *, signal: Path | None = None) -> dict[str, Any]:
    raw = json.dumps(body).encode()
    requested_url = base_url.rstrip("/") + path
    req = urllib.request.Request(
        requested_url,
        data=raw,
        headers={"Content-Type": "application/json", "User-Agent": "sparkrun-glm53-acceptance/1"},
    )
    start = time.time()
    if signal is not None:
        write_signal(signal, "load_started", started_at=start)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
        completed = time.time()
        if signal is not None:
            write_signal(signal, "load_completed", started_at=start, completed_at=completed)
        return {
            "http": resp.status,
            "requested_url": requested_url,
            "effective_url": resp.geturl(),
            "path": path,
            "started_at": start,
            "completed_at": completed,
            "request": body,
            "request_sha256": hashlib.sha256(raw).hexdigest(),
            "response": data,
        }


def post_outcome(base_url: str, path: str, body: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    """Record either a JSON response or a JSON HTTP error without masking it."""
    raw = json.dumps(body).encode()
    requested_url = base_url.rstrip("/") + path
    req = urllib.request.Request(
        requested_url,
        data=raw,
        headers={"Content-Type": "application/json", "User-Agent": "sparkrun-glm53-acceptance/1"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            data = json.load(resp)
            effective_url = resp.geturl()
    except urllib.error.HTTPError as exc:
        status = exc.code
        data = json.loads(exc.read().decode())
        effective_url = exc.geturl()
    return {
        "http": status,
        "requested_url": requested_url,
        "effective_url": effective_url,
        "path": path,
        "started_at": start,
        "completed_at": time.time(),
        "request": body,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "response": data,
    }


def get_json(base_url: str, path: str, timeout: float = 30.0) -> dict[str, Any]:
    start = time.time()
    requested_url = base_url.rstrip("/") + path
    with urllib.request.urlopen(requested_url, timeout=timeout) as resp:
        return {
            "http": resp.status,
            "requested_url": requested_url,
            "effective_url": resp.geturl(),
            "path": path,
            "started_at": start,
            "completed_at": time.time(),
            "response": json.load(resp),
        }


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


OCR_TEXT = "GLM53 OCR 8429"
OCR_GLYPHS = {
    " ": ("00000",) * 7,
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
}


def make_ocr_png() -> bytes:
    width, height, scale, left, top = 544, 96, 6, 14, 27
    pixels = bytearray([255]) * (width * height * 3)
    x0 = left
    for character in OCR_TEXT:
        glyph = OCR_GLYPHS[character]
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "0":
                    continue
                for yy in range(top + gy * scale, top + (gy + 1) * scale):
                    for xx in range(x0 + gx * scale, x0 + (gx + 1) * scale):
                        offset = (yy * width + xx) * 3
                        pixels[offset:offset + 3] = b"\x00\x00\x00"
        x0 += 6 * scale
    rows = [
        b"\x00" + bytes(pixels[y * width * 3:(y + 1) * width * 3])
        for y in range(height)
    ]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
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


def ocr_request(png: bytes) -> dict[str, Any]:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Read the text in this image. Reply with exactly GLM53 OCR 8429."},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }


VIDEO_FIXTURE_ZLIB_B64 = (
    "eNpz93SzsEx8wPCAoZGB4T8DAij+5/ZzDQl2dgxwNdIzYGYECf1kYUgB0jogeZAWBo7/DIwcMh4bFhxs5hDWijmxYeHhdgFlrzkeGxcd7ZYwzrpzYuPi4/0Kzl0ynpuWnJysEbwq5uSmpaenGySfmuO5ednZ2RbFr+6c3Lz8/HyHZi5Zry0rLi72mKwVe2rLysvLAxZ7zfXauurq6ojNWXdPbV19fX3C4S5Z721rbm7OuLwq9vS2tbe3Fzw+Ndd7+7q7uys+v7p7evv6+/sbmLnlfHZseHi4Q1g77syOjY+PT1D2nuezc9PT0zOMs++d2bn5+fkFzt1yvru2vLy8Inh13NldW19f35B8ep7v7m1vb+8ofn3v7O7t7+8faOaW99uz4+PjE5O148/t2fn5+YXF3vP99u76+vrG5uz75/bu/v7+weHuAnn/fXt+fn5xeXX8+X17f3//8Pj0fP/9+/7+/vH59f3z+/f//8/ApvGggUUs40Ejh9qKB008Zi8eNAu4aTxsEQnLeNgqkbbiYZtM2YuH7QptGo86VKZlPOrUWLbiUZfOthePug2OaTzuMbmW8bjX4tmKx302WaICoEhgTGFgREQCSryNRgjdI8QaAKLzwUc="
)


def make_video_fixture() -> bytes:
    return zlib.decompress(base64.b64decode(VIDEO_FIXTURE_ZLIB_B64, validate=True))


def video_rejection_request(gif: bytes) -> dict[str, Any]:
    url = "data:image/gif;base64," + base64.b64encode(gif).decode()
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this video."},
            {"type": "video_url", "video_url": {"url": url}},
        ]}],
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def telemetry_load_request() -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Output the letter A repeatedly until the token budget is exhausted.",
            }
        ],
        "temperature": 0,
        "max_tokens": 512,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def remote_media_request() -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://127.0.0.1:8000/v1/models"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def reasoning_open_stop_request() -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": "Compute 12 times 12. If the result is 144, answer with exactly GLM53_REASONING_STOP_OK and nothing else.",
        }],
        "stop": ["144"],
        "temperature": 0,
        "max_tokens": 2300,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def thinking_disabled_stop_request() -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply exactly: BEFORE Question: AFTER"}],
        "stop": ["Question:"],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct-url", default="http://127.0.0.1:8000")
    ap.add_argument("--proxy-url", default="")
    ap.add_argument("--recipe", default="")
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--telemetry-signal", type=Path, required=True)
    ap.add_argument("--long-repeat", type=int, default=55000)
    args = ap.parse_args()
    if re.fullmatch(r"[0-9a-f]{16}", args.run_id) is None:
        raise ValueError("--run-id must be exactly 16 lowercase hexadecimal characters")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wait_for_collector(args.telemetry_signal)
    fixture = make_quadrant_png()
    fixture_path = out_path.with_name("vision-quadrants.png")
    fixture_path.write_bytes(fixture)
    ocr_fixture = make_ocr_png()
    ocr_fixture_path = Path(__file__).with_name("vision-ocr.png")
    if not ocr_fixture_path.is_file() or ocr_fixture_path.read_bytes() != ocr_fixture:
        raise RuntimeError("committed OCR fixture bytes do not match the canonical generator")
    video_fixture = make_video_fixture()
    video_fixture_path = Path(__file__).with_name("video-tiny.gif")
    if not video_fixture_path.is_file() or video_fixture_path.read_bytes() != video_fixture:
        raise RuntimeError("committed video fixture bytes do not match the canonical generator")
    record: dict[str, Any] = {
        "schema": 1,
        "invocation_argv": sys.argv,
        "run_id": args.run_id,
        "started_at": time.time(),
        "model": MODEL,
        "cluster_id": args.cluster_id,
        "process_role": "exact_final",
        "direct_url": args.direct_url,
        "proxy_url": args.proxy_url or None,
        "recipe_sha256": hashlib.sha256(Path(args.recipe).read_bytes()).hexdigest() if args.recipe else None,
        "acceptance_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "telemetry_capture_script_sha256": hashlib.sha256(
            Path(__file__).with_name("capture_load_telemetry.py").read_bytes()
        ).hexdigest(),
        "fixture": {
            "path": fixture_path.name,
            "sha256": hashlib.sha256(fixture).hexdigest(),
            "size": len(fixture),
            "description": "256x256 RGB PNG: red TL, green TR, blue BL, yellow BR",
        },
        "ocr_fixture": {
            "path": ocr_fixture_path.name,
            "sha256": hashlib.sha256(ocr_fixture).hexdigest(),
            "size": len(ocr_fixture),
            "description": "544x96 RGB PNG with black bitmap text GLM53 OCR 8429",
        },
        "video_fixture": {
            "path": video_fixture_path.name,
            "sha256": hashlib.sha256(video_fixture).hexdigest(),
            "size": len(video_fixture),
            "mime_type": "image/gif",
            "description": "deterministic two-frame 224x224 inline GIF for video=0 rejection",
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
        "results": [
            {"passed": message_text(row) == marker, **row}
            for row, marker in zip(concurrent_rows, markers)
        ],
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

    telemetry_load = post(
        args.direct_url,
        "/v1/chat/completions",
        telemetry_load_request(),
        timeout=900,
        signal=args.telemetry_signal,
    )
    record["checks"]["direct_telemetry_load"] = {
        "passed": telemetry_load.get("response", {}).get("usage", {}).get("completion_tokens") == 512,
        **telemetry_load,
    }

    reasoning_stop = post(
        args.direct_url,
        "/v1/chat/completions",
        reasoning_open_stop_request(),
        timeout=900,
    )
    reasoning_choice = (reasoning_stop.get("response", {}).get("choices") or [{}])[0]
    reasoning_message = reasoning_choice.get("message", {})
    reasoning_text = reasoning_message.get("reasoning") or ""
    reasoning_content = reasoning_message.get("content")
    record["checks"]["direct_reasoning_open_stop"] = {
        "passed": (
            "144" in reasoning_text
            and reasoning_content == "GLM53_REASONING_STOP_OK"
            and reasoning_choice.get("finish_reason") == "stop"
            and reasoning_choice.get("stop_reason") == REASONING_STOP_REASON
        ),
        **reasoning_stop,
    }

    disabled_stop = post(
        args.direct_url,
        "/v1/chat/completions",
        thinking_disabled_stop_request(),
    )
    disabled_choice = (disabled_stop.get("response", {}).get("choices") or [{}])[0]
    disabled_content = disabled_choice.get("message", {}).get("content") or ""
    record["checks"]["direct_thinking_disabled_stop"] = {
        "passed": (
            disabled_content == "BEFORE "
            and disabled_choice.get("finish_reason") == "stop"
            and disabled_choice.get("stop_reason") == "Question:"
        ),
        **disabled_stop,
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
    ocr = post(args.direct_url, "/v1/chat/completions", ocr_request(ocr_fixture), timeout=900)
    record["checks"]["direct_ocr"] = {"passed": message_text(ocr) == OCR_TEXT, **ocr}

    remote_media = post_outcome(
        args.direct_url,
        "/v1/chat/completions",
        remote_media_request(),
    )
    record["checks"]["direct_remote_media_rejected"] = {
        "passed": remote_media.get("http") == 400 and remote_media.get("response") == REMOTE_MEDIA_ERROR,
        **remote_media,
    }
    direct_video = post_outcome(
        args.direct_url,
        "/v1/chat/completions",
        video_rejection_request(video_fixture),
    )
    record["checks"]["direct_video_rejected"] = {
        "passed": direct_video.get("http") == 400 and direct_video.get("response") == VIDEO_LIMIT_ERROR,
        **direct_video,
    }

    if args.proxy_url:
        proxy_models = get_json(args.proxy_url, "/v1/models")
        proxy_model_ok = any(m.get("id") == MODEL for m in proxy_models["response"].get("data", []))
        record["checks"]["proxy_models"] = {"passed": proxy_model_ok, **proxy_models}
        proxy = post(args.proxy_url, "/v1/chat/completions", exact_request("GLM53_PROXY_OK"))
        record["checks"]["proxy_exact"] = {"passed": message_text(proxy) == "GLM53_PROXY_OK", **proxy}
        proxy_vision = post(args.proxy_url, "/v1/chat/completions", vision_request(fixture), timeout=900)
        record["checks"]["proxy_vision"] = {"passed": message_text(proxy_vision) == "RED", **proxy_vision}
        proxy_ocr = post(args.proxy_url, "/v1/chat/completions", ocr_request(ocr_fixture), timeout=900)
        record["checks"]["proxy_ocr"] = {"passed": message_text(proxy_ocr) == OCR_TEXT, **proxy_ocr}
        proxy_remote_media = post_outcome(
            args.proxy_url,
            "/v1/chat/completions",
            remote_media_request(),
        )
        record["checks"]["proxy_remote_media_rejected"] = {
            "passed": (
                proxy_remote_media.get("http") == 400
                and proxy_remote_media.get("response") == PROXY_REMOTE_MEDIA_ERROR
            ),
            **proxy_remote_media,
        }
        proxy_video = post_outcome(
            args.proxy_url,
            "/v1/chat/completions",
            video_rejection_request(video_fixture),
        )
        record["checks"]["proxy_video_rejected"] = {
            "passed": (
                proxy_video.get("http") == 400
                and proxy_video.get("response") == PROXY_VIDEO_LIMIT_ERROR
            ),
            **proxy_video,
        }

    record["completed_at"] = time.time()
    record["passed"] = all(check.get("passed") is True for check in record["checks"].values())
    write_signal(args.telemetry_signal, "acceptance_completed", completed_at=record["completed_at"])
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v["passed"] for k, v in record["checks"].items()}, indent=2))
    print(f"wrote {out_path}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
