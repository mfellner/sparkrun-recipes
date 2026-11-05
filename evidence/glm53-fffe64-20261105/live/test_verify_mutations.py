#!/usr/bin/env python3
"""Mutation controls for live evidence verifier; no live service access."""
import hashlib, json, pathlib, shutil, subprocess, tempfile
SRC=pathlib.Path(__file__).resolve().parent

def refresh(root):
    (root/"SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(root).as_posix()}\n" for p in sorted(root.rglob("*")) if p.is_file() and p.name!="SHA256SUMS"))

def run(root):
    return subprocess.run(["python3",str(root/"verify.py")],capture_output=True,text=True)

with tempfile.TemporaryDirectory() as td:
    clean=pathlib.Path(td)/"clean"; shutil.copytree(SRC,clean)
    assert run(clean).returncode==0
    # Raw bytes drift without index update must fail hash verification.
    bad=pathlib.Path(td)/"bad-hash"; shutil.copytree(SRC,bad)
    with (bad/"api/smoke-results.jsonl").open("a") as f: f.write(" ")
    assert run(bad).returncode != 0
    # Even refreshing every payload hash cannot legitimize a changed canonical route.
    bad=pathlib.Path(td)/"bad-route"; shutil.copytree(SRC,bad)
    f=bad/"api/smoke-results.jsonl"; rows=[json.loads(x) for x in f.read_text().splitlines()]; rows[0]["base"]="http://127.0.0.1:8000"; f.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows)); refresh(bad)
    assert run(bad).returncode != 0
    # A substituted response that preserves an HTTP 200 still fails semantic checking.
    bad=pathlib.Path(td)/"bad-response"; shutil.copytree(SRC,bad)
    f=bad/"api/smoke-results.jsonl"; rows=[json.loads(x) for x in f.read_text().splitlines()]; rows[0]["results"][0]["text"]="unrelated"; f.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows)); refresh(bad)
    assert run(bad).returncode != 0
print("PASS positive and 3 mutation controls (unrefreshed hash, refreshed route substitution, refreshed response substitution)")
