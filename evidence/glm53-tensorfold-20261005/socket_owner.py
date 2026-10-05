#!/usr/bin/env python3
"""Run via `docker exec -i CONTAINER python3 -` to identify the port-8000 owner."""
import json
from pathlib import Path

PORT = 8000
inodes = set()
for family in ("tcp", "tcp6"):
    for line in Path("/proc/net/" + family).read_text().splitlines()[1:]:
        fields = line.split()
        if int(fields[1].split(":")[1], 16) == PORT and fields[3] == "0A":
            inodes.add(fields[9])
assert len(inodes) == 1, f"expected one listening socket: {inodes}"
found = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        fds = [entry.name for entry in (proc / "fd").iterdir()
               if entry.is_symlink() and str(entry.readlink()) == f"socket:[{next(iter(inodes))}]"
               ]
        if fds:
            stat = (proc / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            found.append({"pid": int(proc.name), "start_ticks": int(fields[19]),
                          "cmdline": (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip(),
                          "fds": fds})
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        continue
assert len(found) == 1 and "/usr/local/bin/tensorfold serve " in found[0]["cmdline"], found
print(json.dumps({"port": PORT, "inode": next(iter(inodes)), "owner": found[0]}, sort_keys=True))
