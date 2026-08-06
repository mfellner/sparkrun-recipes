#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.request


def one(base_url, prompt, max_tokens):
    payload = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    submitted = time.perf_counter()
    first_content = None
    last_content = None
    usage = None
    pieces = []
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    now = time.perf_counter()
                    if first_content is None:
                        first_content = now
                    last_content = now
                    pieces.append(content)
    ended = time.perf_counter()
    if first_content is None or last_content is None:
        raise RuntimeError("no non-empty streamed content delta")
    if not usage or not usage.get("completion_tokens"):
        raise RuntimeError(f"missing completion token usage: {usage!r}")
    tokens = int(usage["completion_tokens"])
    decode_seconds = max(last_content - first_content, 1e-9)
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": tokens,
        "ttft_ms": (first_content - submitted) * 1000,
        "decode_tps": (tokens - 1) / decode_seconds if tokens > 1 else 0.0,
        "e2e_output_tps": tokens / (ended - submitted),
        "duration_s": ended - submitted,
        "output_prefix": "".join(pieces)[:120],
    }


def summary(rows):
    result = {"runs": rows}
    for key in ("ttft_ms", "decode_tps", "e2e_output_tps", "duration_s"):
        vals = [float(row[key]) for row in rows]
        result[key] = {
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-tokens", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--mixed-runs", type=int, default=3)
    parser.add_argument("--mixed-tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    structured = (
        "Write only valid minified JSON, with no markdown or commentary. "
        "Generate an object containing an items array. Every item must have "
        "integer id, string name, string category, boolean active, numeric score, "
        "and a short code-like expression. Continue producing distinct items for "
        "the entire available output budget and keep the JSON structure regular."
    )
    for index in range(args.warmups):
        row = one(args.base_url, structured, args.warmup_tokens)
        print(f"warmup {index + 1}/{args.warmups}: {row['decode_tps']:.3f} tok/s", flush=True)

    rows = []
    for index in range(args.runs):
        row = one(args.base_url, structured, args.tokens)
        rows.append(row)
        print(json.dumps({"run": index + 1, **row}), flush=True)

    mixed_prompt = (
        "Write continuous natural prose explaining how a small engineering team "
        "can plan, implement, test, and operate a reliable distributed service. "
        "Use varied sentences and concrete examples, but no lists, JSON, code, "
        "headings, or repeated template phrases. Continue for the full output budget."
    )
    mixed_rows = []
    for index in range(args.mixed_runs):
        row = one(args.base_url, mixed_prompt, args.mixed_tokens)
        mixed_rows.append(row)
        print(json.dumps({"mixed_run": index + 1, **row}), flush=True)

    report = {
        "base_url": args.base_url,
        "model": "glm-5.2",
        "method": {
            "streaming": True,
            "thinking": False,
            "temperature": 0,
            "ignore_eos": True,
            "warmups": args.warmups,
            "warmup_tokens": args.warmup_tokens,
            "measured_runs": args.runs,
            "measured_tokens": args.tokens,
            "mixed_runs": args.mixed_runs,
            "mixed_tokens": args.mixed_tokens,
            "ttft_definition": "request submission to first non-empty content delta",
            "decode_definition": "(completion_tokens - 1) / (last non-empty content delta - first non-empty content delta)",
        },
        "structured": summary(rows),
        "mixed": summary(mixed_rows),
    }
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["structured"], indent=2), flush=True)
    print(json.dumps(report["mixed"], indent=2), flush=True)


if __name__ == "__main__":
    main()
