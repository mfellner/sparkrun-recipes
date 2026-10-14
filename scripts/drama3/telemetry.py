#!/usr/bin/env python3
"""One-second target GPU/host samples with real UTC epochs. SPDX-License-Identifier: MIT."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    deadline = time.monotonic() + args.seconds
    with args.output.open("w") as file:
        while time.monotonic() < deadline and not args.stop_file.exists():
            started = time.time()
            raw = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu,pstate", "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
            util, clock, power, temperature, pstate = [part.strip() for part in raw.split(",")]
            memory = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemAvailable:", "SwapFree:"))}
            sample = dict(timestamp=started, gpu_util_pct=float(util), sm_clock_mhz=float(clock), power_w=float(power), temperature_c=float(temperature), pstate=pstate, **memory)
            file.write(json.dumps(sample) + "\n")
            file.flush()
            time.sleep(max(0, 1 - (time.time() - started)))


if __name__ == "__main__":
    main()
