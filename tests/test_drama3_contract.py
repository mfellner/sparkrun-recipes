"""CPU contracts; GPU acceptance is collected separately on real GB10."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def contract():
    spec = importlib.util.spec_from_file_location("drama_contract", ROOT / "mods/drama3-gb10/contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "127.0.0.1", "100.97.1.1", "8.8.8.8", "localhost"])
def test_binding_rejects_unintended_interfaces(host):
    with pytest.raises(ValueError):
        contract().validate_binding(host, 8010)


def test_target_binding():
    assert contract().validate_binding("192.168.178.70", 8010) == ("192.168.178.70", 8010)


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "a" * 2001}, {"text": "hi", "max_new_tokens": 0}, {"text": "hi", "reference_id": "../private"}, {"text": "hi", "url": "http://evil"}, {"text": "hi", "streaming": "yes"}, {"text": "hi", "max_new_tokens": True}])
def test_request_fails_closed(payload):
    with pytest.raises(ValueError):
        contract().validate_request(payload)


def test_canonical_defaults():
    value = contract().validate_request({"text": "Hello world."})
    assert value["max_new_tokens"] == 1024
    assert value["streaming"] is False
    assert value["format"] == "wav"


def test_missing_snapshot_rejected(tmp_path):
    with pytest.raises(ValueError):
        contract().verify_snapshot(tmp_path, [])


def test_default_reference_is_public_and_checksum_pinned(tmp_path):
    module = contract()
    assert module.DEFAULT_REFERENCE_SHA256 == "4d2bd83ff0f7fe33491b03193abdf623f1b4cc2d8103972b33638d3607d86dc5"
    with pytest.raises(ValueError):
        module.verify_default_reference(b"tampered")


@pytest.mark.parametrize("content_type,origin", [("text/plain", None), ("application/json", "https://evil.invalid"), ("application/json", "null")])
def test_cross_site_simple_requests_rejected(content_type, origin):
    with pytest.raises(ValueError):
        contract().validate_http_headers(content_type, origin, "192.168.178.70", 8010)


def test_cli_and_same_origin_json_allowed():
    assert contract().validate_http_headers("application/json", None, "192.168.178.70", 8010)
    assert contract().validate_http_headers("application/json; charset=utf-8", "http://192.168.178.70:8010", "192.168.178.70", 8010)
