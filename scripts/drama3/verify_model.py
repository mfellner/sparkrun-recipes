#!/usr/bin/env python3
"""Independently rehash immutable model files inside the accepted final container."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect = json.loads(subprocess.check_output(["ssh", "192.168.178.70", "docker", "inspect", args.container], text=True, timeout=30))[0]
    code = 'import json,os,pathlib,contract; m=json.load(open("/opt/drama/model-manifest.json"));contract.verify_snapshot(pathlib.Path(os.environ["DRAMA_MODEL_DIR"]),m["files"]);print(json.dumps(dict(m,passed=True)))'
    command = ["ssh", "192.168.178.70", "docker", "exec", args.container, "python", "-c", shlex.quote(code)]
    receipt = json.loads(subprocess.check_output(command, text=True, timeout=180))
    receipt.update(host="192.168.178.70", container_id=inspect["Id"], image=inspect["Config"]["Image"], verified_at=time.time(), command=command)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("passed", "host", "container_id", "revision")}))


if __name__ == "__main__":
    main()
