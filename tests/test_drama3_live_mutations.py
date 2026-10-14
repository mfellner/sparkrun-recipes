"""Live evidence adversaries rehash manifests, so checksums alone cannot pass."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/drama3-gb10-20261014"


def verifier():
    spec = importlib.util.spec_from_file_location("drama_live_verify", ROOT / "scripts/drama3/verify_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refresh(directory):
    file = directory / "manifest.json"
    value = json.loads(file.read_text())
    value["files"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in value["files"] if (directory / name).is_file()}
    file.write_text(json.dumps(value))


def test_real_evidence_is_complete():
    import sys
    previous_capture = sys.modules.get("capture_runtime")
    result = verifier().verify(EVIDENCE)
    assert result["passed"] is True
    assert result["requests"] == 11
    assert sys.modules.get("capture_runtime") is previous_capture


@pytest.mark.parametrize("kind", ["status", "model", "request", "image", "timestamp", "audio", "receipt-delete", "private-header", "encoded-header", "process", "privileged", "launch-raw", "kernel-drop", "graph-mode", "asr", "stream-delivery", "response-error", "reference-input", "platform-host", "model-instance", "restart", "cache-persist"])
def test_refreshed_manifest_mutations_fail_closed(tmp_path, kind):
    directory = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, directory)
    filename = "english-1.json"
    row = json.loads((directory / filename).read_text())
    if kind == "status":
        row["status"] = 500
    elif kind == "model":
        row["model_revision"] = "0" * 40
    elif kind == "request":
        row["request"]["seed"] += 1
    elif kind == "timestamp":
        row["started_at"] += 60
        row["finished_at"] += 60
    elif kind == "audio":
        row["audio"]["rms"] *= 2
    elif kind == "response-error":
        row["error"] = {"message": "failed"}
    elif kind == "receipt-delete":
        (directory / filename).unlink()
        refresh(directory)
        with pytest.raises(ValueError):
            verifier().verify(directory)
        return
    elif kind == "private-header":
        row["capture"] = '{"headers":{"Authorization":"Bearer test-only-fixture"}}'
    elif kind == "encoded-header":
        row["capture"] = "%257B%2522Authorization%2522%253A%2522Bearer%2520test-only-fixture%2522%257D"
    elif kind in ("process", "privileged", "graph-mode"):
        filename = "runtime-before.json"
        row = json.loads((directory / filename).read_text())
        if kind == "process":
            row["listener"]["pid"] += 1
        elif kind == "privileged":
            row["container"]["privileged"] = True
        else:
            row["health"]["cuda_graph"] = "0"
    elif kind == "launch-raw":
        filename = "launch-state.json"
        row = json.loads((directory / filename).read_text())
        row["status_meta"]["recipe_state"]["_raw"]["env"]["FISH_DRAMA3_FP8_EXPERTS"] = "0"
    elif kind == "asr":
        filename = "asr.json"
        row = json.loads((directory / filename).read_text())
        row["rows"][0]["text"] = "A completely different sentence."
    elif kind == "stream-delivery":
        filename = "streaming.json"
        row = json.loads((directory / filename).read_text())
        row["first_audio_s"] = row["wall_s"]
    elif kind == "image":
        filename = "manifest.json"
        row = json.loads((directory / filename).read_text())
        row["image"] = "ghcr.io/mfellner/unreviewed@sha256:" + "0" * 64
    elif kind == "reference-input":
        shutil.copyfile(directory / "english-1.wav", directory / "reference-input.wav")
        filename = "reference-provenance.json"
        row = json.loads((directory / filename).read_text())
        row["sha256"] = hashlib.sha256((directory / "reference-input.wav").read_bytes()).hexdigest()
    elif kind == "platform-host":
        filename = "platform.json"
        row = json.loads((directory / filename).read_text())
        row["host"] = "unqualified-host"
    elif kind == "model-instance":
        filename = "model-verification.json"
        row = json.loads((directory / filename).read_text())
        row["container_id"] = "f" * 64
    elif kind == "restart":
        filename = "restart-before-runtime.json"
        row = json.loads((directory / "runtime-before.json").read_text())
    elif kind == "cache-persist":
        filename = "cache-after-restart.json"
        row = json.loads((directory / filename).read_text())
        row["compiler_files"] = []
    elif kind == "kernel-drop":
        file = directory / "kernel.txt"
        lines = file.read_text().splitlines()
        file.write_text("\n".join(lines[1:]) + "\n")
        refresh(directory)
        with pytest.raises(ValueError):
            verifier().verify(directory)
        return
    (directory / filename).write_text(json.dumps(row))
    refresh(directory)
    with pytest.raises(ValueError):
        verifier().verify(directory)
