"""Evidence must reject missing artifacts and semantic mutations, even after rehashing."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def verifier():
    spec = importlib.util.spec_from_file_location("drama_verify", ROOT / "scripts/drama3/verify_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_evidence_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        verifier().verify(tmp_path)


@pytest.mark.parametrize("change", [{"status": 500}, {"model_revision": "0" * 40}, {"route": "/v1/chat/completions"}, {"request": {"text": "invented"}}, {"audio": {"duration_s": 0}}, {"finished_at": 0}, {"request_id": "../bad"}, {"headers": {"Authorization": "Bearer private"}}])
def test_receipt_mutations_rejected(change):
    with pytest.raises(ValueError):
        verifier().validate_receipt(change, "english-1")


def test_secret_nested_json_rejected():
    with pytest.raises(ValueError):
        verifier().reject_secrets({"capture": '{"headers":{"Authorization":"Bearer private"}}'})


def test_reference_receipt_hashes_sent_input_not_generated_output(tmp_path, monkeypatch):
    import base64
    import hashlib
    import io
    import json
    verifier()  # Initializes the scripts import path.
    spec = importlib.util.spec_from_file_location("drama_acceptance", ROOT / "scripts/drama3/acceptance.py")
    assert spec is not None and spec.loader is not None
    acceptance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acceptance)
    source = (ROOT / "mods/drama3-gb10/default-reference.wav").read_bytes()
    output = (ROOT / "evidence/drama3-gb10-20261014/english-1.wav").read_bytes()
    assert source != output
    (tmp_path / "reference-input.wav").write_bytes(source)
    (tmp_path / "reference.wav").write_bytes(source)
    (tmp_path / "reference.txt").write_text("A legal public reference transcript.")

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Type": "audio/wav", "X-Request-ID": "a" * 32,
                   "X-Model-Revision": "dd64a85cd82e9521f7202610ede97b367fc83946"}

    def respond(request, **kwargs):
        sent = json.loads(request.data)
        assert base64.b64decode(sent["references"][0]["audio"]) == source
        return Response(output)

    monkeypatch.setattr(acceptance.urllib.request, "urlopen", respond)
    receipt = acceptance.run("http://192.168.178.70:8010", tmp_path, "reference")
    assert (tmp_path / "reference-input.wav").read_bytes() == source
    assert (tmp_path / "reference.wav").read_bytes() == output
    assert receipt["request"]["references"][0]["audio"] == "sha256:" + hashlib.sha256(source).hexdigest()
