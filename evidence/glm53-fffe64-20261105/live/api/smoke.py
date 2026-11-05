#!/usr/bin/env python3
"""Small semantic/concurrent live smoke; no long-running benchmark."""
import concurrent.futures, json, time, urllib.request

MODEL = "GLM-5.3-Flash-EXL3"
ENDPOINTS = {
    "direct": "http://192.168.178.47:8000",
    "proxy": "http://192.168.178.47:4000",
}

def post(base, i):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": f"Reply exactly SMOKE-{i}"}], "max_tokens": 512, "temperature": 0}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            result = json.load(response)
        text = result["choices"][0]["message"]["content"]
        return {"index": i, "http": 200, "ok": f"SMOKE-{i}" in text, "text": text, "elapsed_s": round(time.monotonic()-start, 3), "usage": result.get("usage")}
    except Exception as e:
        return {"index": i, "http": None, "ok": False, "error": repr(e), "elapsed_s": round(time.monotonic()-start, 3)}

for name, base in ENDPOINTS.items():
    print(json.dumps({"endpoint": name, "base": base, "results": [post(base, 0)]}, sort_keys=True))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda i: post(base, i), (1, 2, 3)))
    print(json.dumps({"endpoint": name, "concurrent": 3, "results": results, "passed": sum(r["ok"] for r in results)}, sort_keys=True))
