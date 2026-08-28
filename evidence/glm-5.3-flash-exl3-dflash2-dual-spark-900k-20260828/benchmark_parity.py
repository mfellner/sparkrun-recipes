#!/usr/bin/env python3
"""Repeated GLM-5.3 EXL3 DFlash2 parity benchmark.

Protocol matches sparkDash 1.8.5 (commit e93fc87) and the upstream GLM
repository's 2026-08-28 decode table:
- Structured count 1→200 prompt by default
- temperature=0, top_p=1, thinking disabled
- 400 forced completion tokens, streaming usage enabled
- per-stream decode TPS = (completion_tokens - 1) / (last_visible - first_visible)
- aggregate TPS = sum(per-stream decode tokens) /
  (latest last_visible - earliest first_visible)
- one or more warm-up waves are preserved before measured waves

Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

STRUCTURED_PROMPT = (
    "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
)
CODE_PROMPT = (
    "Output only Python source code. No comments, no docstrings, no markdown fences. "
    "Write functions clamp_00 through clamp_49. Each function is exactly:\n"
    "def clamp_NN(x, lo=0, hi=1):\n"
    "    if x < lo:\n"
    "        return lo\n"
    "    if x > hi:\n"
    "        return hi\n"
    "    return x\n"
    "Change only the function name suffix (00, 01, … 49). One blank line between "
    "functions. No other text."
)
PROSE_PROMPT = (
    "Write a detailed step-by-step explanation of how a hash map works, including "
    "collision handling, resizing, and time complexity. Be thorough."
)
PROMPTS = {"structured": STRUCTURED_PROMPT, "code": CODE_PROMPT, "prose": PROSE_PROMPT}
VISIBLE_KEYS = ("content", "reasoning", "reasoning_content")
SPEC_RE = re.compile(r"^(vllm:spec_decode_[A-Za-z0-9_]+)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)$")


def request_json(url: str, body: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None if body is None else json.dumps(body).encode()
    headers = {"User-Agent": "sparkrun-glm53-parity/1"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def metric_snapshot(base_url: str) -> dict[str, float]:
    req = urllib.request.Request(base_url.rstrip("/") + "/metrics")
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode("utf-8", "replace")
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = SPEC_RE.match(line)
        if not m or m.group(1).endswith("_created"):
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        if "per_pos" in name:
            pm = re.search(r'position="(\d+)"', labels)
            if pm:
                name = f"{name}:position={pm.group(1)}"
        out[name] = out.get(name, 0.0) + value
    return out


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    keys = set(before) | set(after)
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in sorted(keys)}

    def first_suffix(suffix: str) -> float:
        return sum(v for k, v in delta.items() if k.endswith(suffix))

    drafts = first_suffix("num_drafts_total")
    draft_tokens = first_suffix("num_draft_tokens_total")
    accepted = first_suffix("num_accepted_tokens_total")
    return {
        "raw": delta,
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / draft_tokens if draft_tokens else None,
        "accepted_per_draft": accepted / drafts if drafts else None,
    }


def stream_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    barrier: threading.Barrier,
    timeout: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "stop": [],
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "sparkrun-glm53-parity/1"},
    )
    barrier.wait(timeout=30)
    wall_start = time.time()
    perf_start = time.perf_counter()
    first_perf = last_perf = None
    first_wall = last_wall = None
    usage = None
    finish_reason = None
    completion_id = None
    pieces: list[str] = []
    http_status = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http_status = resp.status
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == b"[DONE]":
                    continue
                obj = json.loads(payload)
                completion_id = completion_id or obj.get("id")
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    visible = ""
                    for key in VISIBLE_KEYS:
                        value = delta.get(key)
                        if isinstance(value, str) and value:
                            visible += value
                    if not visible and isinstance(choice.get("text"), str):
                        visible = choice["text"]
                    if visible:
                        now_perf = time.perf_counter()
                        now_wall = time.time()
                        if first_perf is None:
                            first_perf, first_wall = now_perf, now_wall
                        last_perf, last_wall = now_perf, now_wall
                        pieces.append(visible)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    except Exception as exc:  # preserved as a failed stream, never a fabricated row
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "request": body,
            "wall_start": wall_start,
            "wall_end": time.time(),
        }
    wall_end = time.time()
    perf_end = time.perf_counter()
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    decode_tokens = max(completion_tokens - 1, 0)
    decode_seconds = None if first_perf is None or last_perf is None else last_perf - first_perf
    decode_tps = (
        decode_tokens / decode_seconds
        if decode_seconds is not None and decode_seconds > 0 and decode_tokens > 0
        else None
    )
    text = "".join(pieces)
    return {
        "ok": bool(http_status == 200 and completion_tokens > 0 and first_perf is not None),
        "http": http_status,
        "completion_id": completion_id,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tokens": decode_tokens,
        "ttft_s": None if first_perf is None else first_perf - perf_start,
        "decode_s": decode_seconds,
        "decode_tps": decode_tps,
        "end_to_end_output_tps": completion_tokens / (perf_end - perf_start),
        "wall_start": wall_start,
        "first_visible_wall": first_wall,
        "last_visible_wall": last_wall,
        "wall_end": wall_end,
        "request": body,
        "usage": usage,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def run_wave(base_url: str, model: str, prompt_type: str, concurrency: int, max_tokens: int, timeout: float) -> dict[str, Any]:
    base_prompt = PROMPTS[prompt_type]
    prompts = (
        [base_prompt]
        if concurrency == 1
        else [f"{base_prompt} (stream {i + 1}/{concurrency})" for i in range(concurrency)]
    )
    barrier = threading.Barrier(concurrency)
    metrics_before = metric_snapshot(base_url)
    wave_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(stream_request, base_url, model, prompt, max_tokens, barrier, timeout)
            for prompt in prompts
        ]
        streams = [f.result() for f in futures]
    wave_end = time.time()
    metrics_after = metric_snapshot(base_url)
    ok = [s for s in streams if s.get("ok") and s.get("decode_tps") is not None]
    firsts = [s["first_visible_wall"] for s in ok if s.get("first_visible_wall") is not None]
    lasts = [s["last_visible_wall"] for s in ok if s.get("last_visible_wall") is not None]
    aggregate_tps = None
    if firsts and lasts and max(lasts) > min(firsts):
        aggregate_tps = sum(s["decode_tokens"] for s in ok) / (max(lasts) - min(firsts))
    return {
        "concurrency": concurrency,
        "prompt_type": prompt_type,
        "prompts": prompts,
        "wave_start": wave_start,
        "wave_end": wave_end,
        "streams_ok": len(ok),
        "streams_failed": concurrency - len(ok),
        "mean_stream_decode_tps": statistics.mean(s["decode_tps"] for s in ok) if ok else None,
        "median_stream_decode_tps": statistics.median(s["decode_tps"] for s in ok) if ok else None,
        "aggregate_decode_tps": aggregate_tps,
        "mean_ttft_s": statistics.mean(s["ttft_s"] for s in ok) if ok else None,
        "median_ttft_s": statistics.median(s["ttft_s"] for s in ok) if ok else None,
        "speculative_metrics": metric_delta(metrics_before, metrics_after),
        "streams": streams,
    }


def finite(values: list[float | None]) -> list[float]:
    return [x for x in values if x is not None and math.isfinite(x)]


def summarize(measured: list[dict[str, Any]]) -> dict[str, Any]:
    by_c: dict[int, list[dict[str, Any]]] = {}
    for wave in measured:
        by_c.setdefault(int(wave["concurrency"]), []).append(wave)
    out: dict[str, Any] = {}
    for concurrency, rows in sorted(by_c.items()):
        stream = finite([r["mean_stream_decode_tps"] for r in rows])
        aggregate = finite([r["aggregate_decode_tps"] for r in rows])
        ttft = finite([r["mean_ttft_s"] for r in rows])
        out[str(concurrency)] = {
            "measured_waves": len(rows),
            "all_streams_succeeded": all(r["streams_failed"] == 0 for r in rows),
            "mean_stream_decode_tps_median": statistics.median(stream) if stream else None,
            "mean_stream_decode_tps_range": [min(stream), max(stream)] if stream else None,
            "aggregate_decode_tps_median": statistics.median(aggregate) if aggregate else None,
            "aggregate_decode_tps_range": [min(aggregate), max(aggregate)] if aggregate else None,
            "mean_ttft_s_median": statistics.median(ttft) if ttft else None,
            "mean_ttft_s_range": [min(ttft), max(ttft)] if ttft else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--prompt-type", choices=sorted(PROMPTS), default="structured")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--recipe", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = request_json(args.base_url.rstrip("/") + "/v1/models")
    assert any(m.get("id") == args.model for m in models.get("data", [])), models
    script_path = Path(__file__).resolve()
    recipe_sha = None
    if args.recipe:
        recipe_sha = hashlib.sha256(Path(args.recipe).read_bytes()).hexdigest()
    record: dict[str, Any] = {
        "schema": 1,
        "run_id": hashlib.sha256(f"{time.time_ns()}:{script_path}".encode()).hexdigest()[:16],
        "started_at": time.time(),
        "base_url": args.base_url,
        "model": args.model,
        "prompt_type": args.prompt_type,
        "max_tokens": args.max_tokens,
        "concurrencies": args.concurrency,
        "warmups_per_shape": args.warmups,
        "measured_waves_per_shape": args.runs,
        "protocol": {
            "temperature": 0,
            "top_p": 1,
            "thinking": False,
            "min_tokens_equals_max_tokens": True,
            "ignore_eos": True,
            "stream": True,
            "usage_source": "stream_options.include_usage",
            "per_stream_decode_formula": "(completion_tokens - 1) / (last_visible - first_visible)",
            "aggregate_decode_formula": "sum(decode_tokens) / (latest_last_visible - earliest_first_visible)",
            "upstream_sparkdash_revision": "e93fc87d54c8699e98b63a764ab260bf9d446c52",
            "upstream_glm_revision": "4676496e8d4622aaeb0675d79eb15ee1f26c1950",
        },
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "recipe_sha256": recipe_sha,
        "models_response": models,
        "warmups": [],
        "measured": [],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for concurrency in args.concurrency:
        for index in range(args.warmups):
            print(f"warmup C{concurrency} {index + 1}/{args.warmups}", flush=True)
            wave = run_wave(args.base_url, args.model, args.prompt_type, concurrency, args.max_tokens, args.timeout)
            wave["kind"] = "warmup"
            record["warmups"].append(wave)
            out_path.write_text(json.dumps(record, indent=2) + "\n")
        for index in range(args.runs):
            print(f"measured C{concurrency} {index + 1}/{args.runs}", flush=True)
            wave = run_wave(args.base_url, args.model, args.prompt_type, concurrency, args.max_tokens, args.timeout)
            wave["kind"] = "measured"
            record["measured"].append(wave)
            record["summary"] = summarize(record["measured"])
            out_path.write_text(json.dumps(record, indent=2) + "\n")
    record["completed_at"] = time.time()
    record["summary"] = summarize(record["measured"])
    record["passed"] = all(v["all_streams_succeeded"] for v in record["summary"].values())
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["summary"], indent=2))
    print(f"wrote {out_path}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
