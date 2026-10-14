#!/usr/bin/env python3
"""Read a small compiler/output persistence witness; no model or private voice content."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def capture(container, request_id):
    if len(request_id) != 32 or any(char not in "0123456789abcdef" for char in request_id):
        raise ValueError("invalid request ID")
    code = '''import os,pathlib,hashlib,json
base=pathlib.Path(os.environ["TRITON_CACHE_DIR"])
files=sorted(p for p in base.rglob("*") if p.is_file())
rows=[dict(path=str(p.relative_to(base)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size) for p in files[:8]]
output=pathlib.Path(os.environ["DRAMA_OUTPUT_DIR"])/"''' + request_id + '''.wav"
print(json.dumps(dict(cache_path=str(base),cache_file_count=len(files),compiler_files=rows,output_path=str(output),output_exists=output.is_file(),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))'''
    output = subprocess.check_output(["ssh", "192.168.178.70", "docker", "exec", container, "python", "-c", shlex.quote(code)], text=True, timeout=30)
    result = json.loads(output)
    result.update(host="192.168.178.70", captured_at=time.time(), container_name=container, request_id=request_id)
    if not result["compiler_files"] or not result["output_exists"]:
        raise ValueError("missing persistent compiler/output witness")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(capture(args.container, args.request_id), indent=2) + "\n")
    print(args.output)
