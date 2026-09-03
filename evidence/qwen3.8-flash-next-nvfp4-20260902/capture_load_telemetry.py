#!/usr/bin/env python3
"""Capture simultaneous rank telemetry during a bounded workload."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import socket
import subprocess
import time
from pathlib import Path

REMOTE = r'''
import json
import socket
import subprocess
import time
from pathlib import Path

def counters():
    out = {}
    for path in Path("/sys/class/infiniband").glob("*/ports/*/counters/*"):
        if path.name in {"port_xmit_data", "port_rcv_data", "port_xmit_packets", "port_rcv_packets"}:
            try:
                out[str(path)] = int(path.read_text().strip())
            except (OSError, ValueError):
                pass
    return out

count = __COUNT__
interval = __INTERVAL__
before = counters()
rows = []
for _ in range(count):
    started = time.time()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,temperature.gpu,pstate", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    rows.append({"unix": started, "nvidia_smi": result.stdout.strip()})
    delay = interval - (time.time() - started)
    if delay > 0:
        time.sleep(delay)
after = counters()
print(json.dumps({
    "host": socket.gethostname(),
    "before": before,
    "after": after,
    "delta": {key: after.get(key, 0) - value for key, value in before.items()},
    "samples": rows,
}, sort_keys=True))
'''


def capture(host: str, count: int, interval: float, via: str | None) -> dict:
    code = REMOTE.replace("__COUNT__", str(count)).replace("__INTERVAL__", repr(interval))
    ssh_options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if via and host != via:
        command = ["ssh", *ssh_options, via, "ssh", *ssh_options, host, "python3", "-"]
    else:
        command = ["ssh", *ssh_options, host, "python3", "-"]
    errors: list[str] = []
    for attempt in range(3):
        result = subprocess.run(
            command,
            input=code,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=count * interval + 60,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        errors.append(result.stderr.strip())
        if attempt < 2:
            time.sleep(2**attempt)
    raise RuntimeError(f"telemetry SSH failed for {host}: {' | '.join(errors)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", nargs="+", required=True)
    parser.add_argument("--via", help="SSH controller used to reach non-controller ranks")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
        rows = list(pool.map(lambda host: capture(host, args.count, args.interval, args.via), args.hosts))
    payload = {
        "capture_host": socket.gethostname(),
        "started_unix": started,
        "finished_unix": time.time(),
        "hosts": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
