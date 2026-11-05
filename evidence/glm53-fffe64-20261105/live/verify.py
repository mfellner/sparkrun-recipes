#!/usr/bin/env python3
"""Fail-closed integrity and bounded live-smoke checks."""
import hashlib, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent

def fail(message):
    print("FAIL " + message); raise SystemExit(1)

# The committed checksum index is authoritative for every payload except itself.
checksum_file = ROOT / "SHA256SUMS"
if not checksum_file.is_file(): fail("missing checksum index")
expected = {}
for line in checksum_file.read_text().splitlines():
    parts = line.split(None, 1)
    if len(parts) != 2: fail("malformed checksum row")
    digest, name = parts[0], parts[1].lstrip("* ")
    if name.startswith("./"): name = name[2:]
    if name in expected or len(digest) != 64: fail("duplicate or malformed checksum row")
    expected[name] = digest
actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and p != checksum_file}
if set(expected) != actual: fail(f"checksum inventory mismatch: missing={sorted(actual-set(expected))}, extra={sorted(set(expected)-actual)}")
for name, digest in expected.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest: fail(f"payload hash mismatch: {name}")

required = ["metadata/cluster-status.json", "metadata/effective-recipe.json", "metadata/runtime-logs.txt", "hosts/rank0.txt", "hosts/rank1.txt", "api/smoke.py", "api/smoke-results.jsonl", "manifest.json", "README.md"]
if any(not (ROOT/n).is_file() or not (ROOT/n).stat().st_size for n in required): fail("missing/empty required evidence")
manifest = json.loads((ROOT/"manifest.json").read_text())
status = json.loads((ROOT/"metadata/cluster-status.json").read_text())
effective = json.loads((ROOT/"metadata/effective-recipe.json").read_text())
job_id = "sparkrun_58201f8c57f8af18_f50636d19f88"
if manifest.get("job_id") != job_id or manifest.get("cluster") != "spark-pair2": fail("manifest launch identity mismatch")
groups = status.get("groups", {})
if set(groups) != {job_id}: fail("cluster status job identity mismatch")
g = groups[job_id]
if g.get("meta", {}).get("cluster_id") != job_id or g.get("meta", {}).get("cluster") != manifest["cluster"]: fail("persisted job metadata identity mismatch")
if g.get("meta", {}).get("tensor_parallel") != 2 or g.get("meta", {}).get("pipeline_parallel") != 1 or len(g.get("containers", [])) != 2: fail("TP/PP/rank identity mismatch")
if {c.get("role") for c in g["containers"]} != {"node_0", "node_1"}: fail("expected exactly two rank containers")
if g.get("meta", {}).get("effective_container_image") != manifest.get("runtime_image"): fail("runtime image does not match persisted launch")
if effective.get("model_revision") != manifest.get("model_revision") or effective.get("container") != manifest.get("runtime_image") or effective.get("metadata",{}).get("source_revision") != manifest.get("serving_revision"): fail("effective recipe identity mismatch")
if effective.get("defaults",{}).get("served_model_name") != "GLM-5.3-Flash-EXL3" or effective.get("defaults",{}).get("tensor_parallel") != 2: fail("effective recipe serving configuration mismatch")
# Requests are canonicalized from fixed reviewed constants, not receipt fields.
model = "GLM-5.3-Flash-EXL3"
endpoints = {"direct":"http://192.168.178.47:8000", "proxy":"http://192.168.178.47:4000"}
rows = [json.loads(x) for x in (ROOT/"api/smoke-results.jsonl").read_text().splitlines()]
if len(rows) != 4: fail("expected four route/wave records")
seen = set()
for row in rows:
    ep = row.get("endpoint")
    wave = "serial" if row.get("concurrent") is None else "concurrent"
    key = (ep, wave)
    if ep not in endpoints or key in seen: fail("unexpected/duplicate endpoint wave")
    seen.add(key)
    if row.get("base", endpoints[ep]) != endpoints[ep]: fail(f"{ep} endpoint base mismatch")
    concurrent = row.get("concurrent")
    vals = row.get("results")
    if not isinstance(vals,list) or (concurrent is None and len(vals)!=1) or (concurrent==3 and len(vals)!=3) or concurrent not in (None,3): fail("canonical request wave/count mismatch")
    expected_indices = [0] if concurrent is None else [1,2,3]
    if [v.get("index") for v in vals] != expected_indices: fail("canonical request indices mismatch")
    for v in vals:
        i=v["index"]
        # smoke.py's request is fixed: role=user, exact prompt, max_tokens=512, temperature=0.
        if v.get("http") != 200 or not v.get("ok") or v.get("error") is not None: fail(f"{ep} request {i} unsuccessful")
        if not isinstance(v.get("text"),str) or f"SMOKE-{i}" not in v["text"]: fail(f"{ep} request {i} lacks expected semantic response")
if seen != {(ep,w) for ep in endpoints for w in ("serial","concurrent")}: fail("missing route/wave")
print("PASS hashes, persisted launch/model/image identity, fixed route/wave schema, and semantic smoke responses verified")
print("NOTE short smoke only; configured 850k is not measured capacity; host OOM event remains un-attributed.")
