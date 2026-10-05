#!/usr/bin/env python3
"""Map a Docker-top host PID to its container namespace and start identity."""
import json
from pathlib import Path
import sys

assert len(sys.argv) == 2 and sys.argv[1].isdigit()
pid = int(sys.argv[1])
proc = Path("/proc") / str(pid)
status = proc.joinpath("status").read_text().splitlines()
ns = [int(v) for line in status if line.startswith("NSpid:") for v in line.split()[1:]]
assert len(ns) == 2 and ns[0] == pid and ns[1] > 1, ns
stat = proc.joinpath("stat").read_text()
start_ticks = int(stat[stat.rfind(")") + 2:].split()[19])
cmdline = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
assert "/usr/local/bin/tensorfold serve " in cmdline and start_ticks > 0
print(json.dumps({"host_pid": pid, "namespace_pids": ns, "start_ticks": start_ticks,
                  "cmdline": cmdline}, sort_keys=True))
