#!/usr/bin/env python3
"""Apply and verify the real patch stack twice in a disposable CPU-only image.

No weights, GPU imports, network, serving processes or launch-time HCA gates.
The CLI refuses unscoped execution; never point it at a serving container.
"""
import hashlib
import os
from pathlib import Path
import re
import runpy
import subprocess


def main():
    if (os.environ.get("GLM53_DISPOSABLE_CPU_TEST") != "1"
            or not Path("/.dockerenv").is_file()
            or list(Path("/dev").glob("nvidia*"))):
        raise SystemExit("requires explicitly opted-in disposable CPU-only container without GPU devices")
    mod = Path(__file__).resolve().parent
    run = (mod / "run.sh").read_text()
    site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
    previous = None
    for repeat in range(2):
        for line in run.splitlines():
            if line.startswith("install "):
                subprocess.run(line, cwd=mod, shell=True, check=True)
            elif re.fullmatch(r"python3 (?:upstream/overlay/)?patch_\w+\.py", line):
                argv = line.split()
                argv.insert(1, "-S")
                subprocess.run(argv, cwd=mod, check=True)
        ns = runpy.run_path(str(mod / "verify_runtime_patch_state.py"))
        failures = ns["verify_runtime"](site, import_runtime=False)
        if failures:
            raise SystemExit("\n".join(failures))
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in site.rglob("*.py")}
        if previous is not None and previous != hashes:
            raise SystemExit("second complete apply changed runtime bytes")
        previous = hashes
        print(f"CPU_PATCH_COMPOSITION_PASS round={repeat + 1}", flush=True)
    print("CPU_PATCH_IDEMPOTENCY_PASS (static source, not live imports/inference)", flush=True)


if __name__ == "__main__":
    main()
