#!/usr/bin/env python3
"""Collect real HTTP-to-WAV acceptance, never manufacture receipts. SPDX-License-Identifier: MIT."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np
import soundfile as sf
from fixtures import request_for


def inspect_audio(raw):
    samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if not np.isfinite(samples).all() or samples.shape[1] != 1 or rate != 44100:
        raise ValueError("unexpected or nonfinite decoded audio")
    duration = len(samples) / rate
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if not 0.5 <= duration <= 180 or rms < 0.001:
        raise ValueError("implausible duration or silent audio")
    return dict(duration_s=duration, sample_rate=rate, channels=1, samples=len(samples), rms=rms, peak=float(np.abs(samples).max()), finite=True)


def run(endpoint, directory, name):
    request = request_for(name)
    reference_raw = b""
    if name == "reference":
        reference_raw = (directory / "reference-input.wav").read_bytes()
        request["references"] = [{"audio": base64.b64encode(reference_raw).decode(), "text": (directory / "reference.txt").read_text().strip(), "consent": True}]
    encoded = json.dumps(request).encode()
    started = time.time()
    start = time.perf_counter()
    parts = []
    chunks = []
    first_audio = None
    with urllib.request.urlopen(urllib.request.Request(endpoint + "/v1/tts", encoded, {"Content-Type": "application/json"}), timeout=1800) as response:
        status = response.status
        headers = {key.lower(): value for key, value in response.headers.items() if key.lower() in {"content-type", "x-request-id", "x-model-revision"}}
        total = 0
        while part := response.read1(65536):
            parts.append(part)
            chunks.append({"elapsed_s": time.perf_counter() - start, "bytes": len(part)})
            total += len(part)
            # First payload sample, NOT WAV header/HTTP metadata.
            if total > 44 and first_audio is None:
                first_audio = time.perf_counter() - start
    wall = time.perf_counter() - start
    raw = b"".join(parts)
    audio = inspect_audio(raw)
    file = directory / (name + ".wav")
    file.write_bytes(raw)
    # Public reference bytes are separately hash-pinned; avoid duplicating base64 in receipts.
    if name == "reference":
        request["references"][0]["audio"] = "sha256:" + hashlib.sha256(reference_raw).hexdigest()
    receipt = dict(name=name, endpoint=endpoint, route="/v1/tts", status=status, headers=headers, model="fishaudio/drama-3", model_revision=headers.get("x-model-revision"), request_id=headers.get("x-request-id"), request=request, started_at=started, finished_at=time.time(), wall_s=wall, first_audio_s=first_audio, chunks=chunks, rtf=wall/audio["duration_s"], audio=audio, audio_file=file.name, audio_sha256=hashlib.sha256(raw).hexdigest())
    (directory / (name + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("name", "request_id", "wall_s", "first_audio_s", "rtf", "audio")}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://192.168.178.70:8010")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", default="english-1,english-2,german-1,german-2,expression,streaming,longer,reference,overlap-a,overlap-b")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = args.names.split(",")
    for name in names:
        if name not in ("overlap-a", "overlap-b"):
            run(args.endpoint, args.output, name)
    if "overlap-a" in names or "overlap-b" in names:
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda name: run(args.endpoint, args.output, name), [name for name in names if name in ("overlap-a", "overlap-b")]))


if __name__ == "__main__":
    main()
