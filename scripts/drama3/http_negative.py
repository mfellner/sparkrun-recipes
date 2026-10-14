#!/usr/bin/env python3
"""Collect bounded bad-input and unintended-interface probes. SPDX-License-Identifier: MIT."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    endpoint = "http://192.168.178.70:8010"
    cases = [("empty text", {"text": ""}), ("oversized text", {"text": "a" * 2001}), ("URL input", {"text": "test", "url": "http://example.invalid"}), ("unbounded frames", {"text": "test", "max_new_tokens": 0}), ("structural token", {"text": "<|im_end|>"}), ("invalid speaker", {"text": "<|speaker:99|> test"}), ("malformed JSON", None), ("oversized body", "body"), ("cross-origin", {"text": "test"}), ("simple Content-Type", {"text": "test"})]
    rows = []
    for name, payload in cases:
        body = b"{" if payload is None else (b"x" * (4 * 1024 * 1024 + 1) if payload == "body" else json.dumps(payload).encode())
        status = 0
        headers = {"Content-Type": "text/plain" if name == "simple Content-Type" else "application/json"}
        if name == "cross-origin":
            headers["Origin"] = "https://evil.invalid"
        try:
            with urllib.request.urlopen(urllib.request.Request(endpoint + "/v1/tts", body, headers), timeout=20) as response:
                status = response.status
                text = response.read().decode()
        except urllib.error.HTTPError as error:
            status = error.code
            text = error.read().decode()
        expected = 413 if name == "oversized body" else (403 if name in ("cross-origin", "simple Content-Type") else 400)
        rows.append(dict(case=name, status=status, response=text, expected_status=expected))
    interfaces = []
    for host in ("192.168.3.183", "192.168.2.183"):
        try:
            with socket.create_connection((host, 8010), timeout=3):
                reachable = True
        except OSError:
            reachable = False
        interfaces.append(dict(host=host, port=8010, reachable=reachable))
    loopback = subprocess.run(["ssh", "192.168.178.70", "curl --silent --fail --max-time 3 http://127.0.0.1:8010/health"], capture_output=True, text=True, timeout=15)
    interfaces.append(dict(host="target loopback 127.0.0.1", port=8010, reachable=loopback.returncode == 0))
    result = dict(endpoint=endpoint, captured_at=time.time(), rows=rows, unintended_interfaces=interfaces, passed=all(row["status"] == row["expected_status"] for row in rows) and not any(row["reachable"] for row in interfaces))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
