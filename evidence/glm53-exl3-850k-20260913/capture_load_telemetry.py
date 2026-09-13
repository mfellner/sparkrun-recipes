#!/usr/bin/env python3
"""Capture before/during/after telemetry joined to one acceptance load receipt."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HOSTS = ("192.168.178.47", "192.168.178.46")
HCAS = ("rocep1s0f1", "roceP2p1s0f1")
REDACTED = "[REDACTED]"
SENSITIVE_NAME = re.compile(
    r"(?i)(?:^|[_-])(?:authorization|api[_-]?key|access[_-]?token|auth[_-]?token|token|password|passwd|secret|credential|connection[_-]?string)(?:$|[_-])"
)


def redact_text(text: str) -> str:
    output: list[str] = []
    assignment = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*=\s*)(.*)$")
    header = re.compile(r"(?i)^(\s*(?:proxy-)?authorization\s*:\s*).*$")
    inline = re.compile(
        r"(?i)(\b(?:authorization|api[_-]?key|(?:api|auth|access|refresh|hf)[_-]?token|password|passwd|secret|credential|connection[_-]?string)\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|\S+)"
    )
    uri_credentials = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@")
    for line in text.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        match = assignment.fullmatch(body)
        if match and SENSITIVE_NAME.search(match.group(2)):
            body = "".join(match.group(1, 2, 3)) + REDACTED
        elif header.fullmatch(body):
            body = header.sub(r"\1" + REDACTED, body)
        else:
            body = inline.sub(r"\1" + REDACTED, body)
            body = uri_credentials.sub(r"\1" + REDACTED + "@", body)
        output.append(body + ending)
    return "".join(output)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_signal(path: Path, state: str) -> None:
    path.write_text(json.dumps({"state": state, "at": time.time()}) + "\n")


def read_signal(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def wait_state(path: Path, state: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = read_signal(path)
        if row.get("state") == state:
            return row
        time.sleep(0.1)
    raise TimeoutError(f"telemetry signal did not reach {state!r}")


def get_metrics(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"metrics HTTP {response.status}")
        return response.read().decode()


def relevant_metrics(text: str) -> list[str]:
    prefixes = (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:num_requests_waiting_by_reason",
        "vllm:kv_cache_usage_perc",
    )
    return [line for line in text.splitlines() if line.startswith(prefixes)]


def running_requests(lines: list[str]) -> float:
    total = 0.0
    for line in lines:
        if line.startswith("vllm:num_requests_running"):
            total += float(line.rsplit(None, 1)[1])
    return total


def remote_sample(host: str) -> tuple[str, str, str]:
    hca_words = " ".join(shlex.quote(hca) for hca in HCAS)
    command = (
        "for _ in $(seq 1 10); do "
        "nvidia-smi --query-gpu=pstate,utilization.gpu,clocks.sm,power.draw "
        "--format=csv,noheader; sleep 0.2; done; "
        f"for h in {hca_words}; do "
        "printf '%s rate=' \"$h\"; cat /sys/class/infiniband/$h/ports/1/rate; "
        "printf '%s state=' \"$h\"; cat /sys/class/infiniband/$h/ports/1/state; "
        "printf '%s xmit=' \"$h\"; cat /sys/class/infiniband/$h/ports/1/counters/port_xmit_data; "
        "printf '%s rcv=' \"$h\"; cat /sys/class/infiniband/$h/ports/1/counters/port_rcv_data; "
        "done"
    )
    started_at = now_iso()
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "bash", "-lc", shlex.quote(command)],
        text=True,
        capture_output=True,
        timeout=60,
    )
    completed_at = now_iso()
    if result.returncode != 0:
        raise RuntimeError(f"telemetry SSH failed for {host}: {result.stderr.strip()}")
    return result.stdout, started_at, completed_at


def capture(name: str, metrics_url: str, *, require_running: bool = False) -> str:
    deadline = time.monotonic() + 60
    while True:
        metrics = relevant_metrics(get_metrics(metrics_url))
        if not require_running or running_requests(metrics) >= 1:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("direct telemetry load never became active")
        time.sleep(0.1)
    lines = [f"sample={name} time={now_iso()}", *metrics]
    for host in HOSTS:
        sample_text, acquisition_started, acquisition_completed = remote_sample(host)
        lines.append(f"host={host} acquisition_started_at={acquisition_started}")
        lines.extend(sample_text.splitlines())
        lines.append(f"host={host} acquisition_completed_at={acquisition_completed}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    args.signal.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.signal.unlink(missing_ok=True)
    before = capture("before", args.metrics_url)
    write_signal(args.signal, "collector_ready")
    wait_state(args.signal, "load_started", args.timeout)
    during = capture("during", args.metrics_url, require_running=True)
    wait_state(args.signal, "load_completed", args.timeout)
    wait_state(args.signal, "acceptance_completed", args.timeout)
    after = capture("after", args.metrics_url)
    args.out.write_text(redact_text(before + during + after))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
