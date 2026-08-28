#!/usr/bin/env python3
"""Timestamped all-rank GPU and HCA telemetry sampler for parity runs."""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STOP = threading.Event()
REMOTE = r'''set -eu
printf 'gpu='
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,pstate,clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits 2>&1 || true
for d in /sys/class/infiniband/*; do
  [ -d "$d" ] || continue
  h=$(basename "$d")
  for c in port_xmit_data port_rcv_data port_xmit_packets port_rcv_packets port_xmit_discards port_rcv_errors; do
    f="$d/ports/1/counters/$c"
    [ -r "$f" ] && printf 'hca.%s.%s=%s\n' "$h" "$c" "$(cat "$f")"
  done
done
'''


def collect(host: str) -> dict:
    start = time.time()
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, REMOTE],
        text=True,
        capture_output=True,
        timeout=20,
    )
    return {
        "sample_started_at": start,
        "sample_completed_at": time.time(),
        "host": host,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", nargs="+", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)

    def stop(_signum, _frame):
        STOP.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with path.open("w", buffering=1) as fh, ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
        fh.write(json.dumps({"kind": "metadata", "started_at": time.time(), "hosts": args.hosts, "interval_s": args.interval}) + "\n")
        while not STOP.is_set():
            tick = time.time()
            futures = [pool.submit(collect, host) for host in args.hosts]
            for future in futures:
                try:
                    fh.write(json.dumps({"kind": "sample", **future.result()}) + "\n")
                except Exception as exc:
                    fh.write(json.dumps({"kind": "sample_error", "completed_at": time.time(), "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            STOP.wait(max(0.0, args.interval - (time.time() - tick)))
        fh.write(json.dumps({"kind": "metadata_end", "completed_at": time.time()}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
