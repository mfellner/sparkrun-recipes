#!/usr/bin/env python3
"""Check that existing GLM/Qwen direct and proxy routes still generate."""

import json
import pathlib
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

CASES = [
    ("glm_direct", "http://127.0.0.1:8000/v1/chat/completions", "GLM-5.3-Flash-EXL3", "GLM-ALIVE-531"),
    ("qwen_direct", "http://192.168.178.48:8000/v1/chat/completions", "qwen3.8-flash-next", "QWEN-ALIVE-381"),
    ("glm_proxy", "http://192.168.178.47:4000/v1/chat/completions", "GLM-5.3-Flash-EXL3", "GLM-PROXY-531"),
    ("qwen_proxy", "http://192.168.178.47:4000/v1/chat/completions", "qwen3.8-flash-next", "QWEN-PROXY-381"),
]


def request(url, model, marker):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Reply with exactly {marker} and nothing else."}],
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            body = json.loads(response.read())
            message = body["choices"][0]["message"]
            content = message.get("content") or ""
            return {
                "status": response.status,
                "content": content,
                "exact_match": content.strip() == marker,
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": body.get("usage", {}),
                "elapsed_s": time.monotonic() - started,
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "error": exc.read().decode(errors="replace"),
            "exact_match": False,
            "elapsed_s": time.monotonic() - started,
        }


def main():
    output = pathlib.Path(__file__).with_name("non-regression-result.json")
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(uuid.uuid4()),
        "checks": {},
    }
    for name, url, model, marker in CASES:
        result["checks"][name] = request(url, model, marker)
    result["pass"] = all(item["status"] == 200 and item["exact_match"] for item in result["checks"].values())
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
