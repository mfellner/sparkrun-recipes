"""Fail-closed launch and request contracts (stdlib-only). SPDX-License-Identifier: MIT."""
import base64
import hashlib
import ipaddress
from pathlib import Path

MODEL_ID = "fishaudio/drama-3"
MODEL_REVISION = "dd64a85cd82e9521f7202610ede97b367fc83946"
SOURCE_REVISION = "1ff3c3ed2f80d3106a83909dd25b83129c3d816f"
DEFAULT_REFERENCE_SHA256 = "4d2bd83ff0f7fe33491b03193abdf623f1b4cc2d8103972b33638d3607d86dc5"
MAX_BODY = 4 * 1024 * 1024


def verify_default_reference(audio):
    if hashlib.sha256(audio).hexdigest() != DEFAULT_REFERENCE_SHA256:
        raise ValueError("default public-domain reference checksum mismatch")
    return True


def validate_binding(host, port):
    address = ipaddress.IPv4Address(host)
    allowed = any(address in ipaddress.IPv4Network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
    if not allowed or type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("bind requires an explicit RFC1918 IPv4 and unprivileged port")
    return str(address), port


def validate_http_headers(content_type, origin, host, port):
    if content_type.split(";", 1)[0].strip().casefold() != "application/json":
        raise ValueError("JSON Content-Type required; browser simple requests are denied")
    if origin is not None and origin != f"http://{host}:{port}":
        raise ValueError("cross-origin browser requests are denied")
    return True


def validate_request(payload):
    defaults = dict(format="wav", streaming=False, max_new_tokens=1024, seed=42, temperature=0.7, top_p=0.7, chunk_length=200, single_chunk=False, references=[])
    if not isinstance(payload, dict) or set(payload) - (set(defaults) | {"text"}):
        raise ValueError("unknown fields; URL and path inputs are not supported")
    value = {**defaults, **payload}
    text = value.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError("text must contain 1..2000 characters")
    if value["format"] != "wav" or any(type(value[k]) is not bool for k in ("streaming", "single_chunk")):
        raise ValueError("only WAV with boolean streaming/single_chunk is supported")
    for key, low, high in (("max_new_tokens", 1, 1024), ("seed", 0, 2147483647), ("chunk_length", 100, 1000)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"invalid {key}")
    for key in ("temperature", "top_p"):
        if type(value[key]) not in (int, float) or not 0.1 <= value[key] <= 1.0:
            raise ValueError(f"invalid {key}")
    refs = value["references"]
    if not isinstance(refs, list) or len(refs) > 1:
        raise ValueError("at most one reference is supported")
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"audio", "text", "consent"} or ref["consent"] is not True:
            raise ValueError("reference requires audio, transcript and affirmative consent/legal right")
        if not isinstance(ref["text"], str) or not 1 <= len(ref["text"]) <= 1000 or not isinstance(ref["audio"], str) or len(ref["audio"]) > 3000000:
            raise ValueError("invalid reference size or transcript")
        try:
            audio = base64.b64decode(ref["audio"], validate=True)
        except Exception as exc:
            raise ValueError("reference must be base64 WAV") from exc
        if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
            raise ValueError("reference must be WAV")
    return value


def verify_snapshot(path, entries):
    path = Path(path)
    required = {"config.json", "tokenizer.json", "model.safetensors.index.json", "codec.pth"} | {f"model-{i:05d}-of-00008.safetensors" for i in range(1, 9)}
    if not required <= {entry["path"] for entry in entries}:
        raise ValueError("missing required model manifest entries")
    for entry in entries:
        name = entry["path"]
        if Path(name).name != name:
            raise ValueError("invalid manifest path")
        file = path / name
        if not file.is_file() or file.stat().st_size != entry["size"]:
            raise ValueError(f"missing/wrong-size model file: {name}")
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"model digest mismatch: {name}")
    return True
