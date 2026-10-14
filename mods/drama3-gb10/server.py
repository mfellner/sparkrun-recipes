"""Narrow native Fish adapter: bounded WAV API, serialized GPU access. SPDX-License-Identifier: MIT."""
import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from contract import MAX_BODY, MODEL_ID, MODEL_REVISION, SOURCE_REVISION, DEFAULT_REFERENCE_SHA256, validate_binding, validate_request, verify_snapshot, verify_default_reference, validate_http_headers
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.inference_engine.utils import wav_chunk_header
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue, load_codec_model
from fish_speech.models.text2semantic.fish_omni.generate import validate_drama3_text
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

LOCK = threading.Lock()
SLOTS = threading.BoundedSemaphore(3)
ENGINE = None
DEFAULT_REFERENCE = None
STARTED = time.time()


class Engine(TTSInferenceEngine):
    def load_audio(self, reference_audio, sr):
        # Torchaudio 2.13 no longer implements the upstream backend='soundfile' API.
        # Decode only bounded inline WAV with soundfile, then use native resampling.
        with sf.SoundFile(io.BytesIO(reference_audio)) as file:
            if file.format != "WAV" or file.channels > 2 or not 8000 <= file.samplerate <= 96000 or not 0.5 <= len(file) / file.samplerate <= 30:
                raise ValueError("reference must be 0.5..30s mono/stereo WAV at 8..96kHz")
            samples = file.read(dtype="float32", always_2d=True).mean(axis=1)
            source_rate = file.samplerate
        if not np.isfinite(samples).all():
            raise ValueError("nonfinite reference audio")
        if source_rate != sr:
            samples = torchaudio.functional.resample(torch.from_numpy(samples), source_rate, sr).numpy()
        return samples


def native_request(value):
    refs = [ServeReferenceAudio(audio=base64.b64decode(ref["audio"], validate=True), text=ref["text"], speaker=0) for ref in value["references"]]
    if not refs:
        refs = [DEFAULT_REFERENCE]
    return ServeTTSRequest(**{k: v for k, v in value.items() if k != "references"}, references=refs, use_memory_cache="off", normalize=False)


def synthesize(value, rid, messages):
    entered = time.time()
    try:
        with LOCK:
            begin = time.time()
            torch.cuda.reset_peak_memory_stats()
            final = None
            rate = ENGINE.decoder_model.sample_rate
            for result in ENGINE.inference(native_request(value)):
                if result.code == "error":
                    raise RuntimeError(str(result.error))
                if result.code == "segment" and value["streaming"]:
                    samples = result.audio[1]
                    if not np.isfinite(samples).all():
                        raise RuntimeError("nonfinite generated audio")
                    messages.put(np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes())
                if result.code == "final":
                    rate, final = result.audio
            if final is None or not np.isfinite(final).all() or final.size < rate // 4 or float(np.sqrt(np.mean(final.astype(np.float64)**2))) < 0.001:
                raise RuntimeError("empty, nonfinite or silent synthesis")
            output = Path(os.environ["DRAMA_OUTPUT_DIR"]) / (rid + ".wav")
            sf.write(output, final, rate, subtype="PCM_16")
            if not value["streaming"]:
                messages.put(output.read_bytes())
            receipt = dict(request_id=rid, queued_at=entered, gpu_started_at=begin, finished_at=time.time(), text_sha256=hashlib.sha256(value["text"].encode()).hexdigest(), audio_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), duration_s=len(final)/rate, peak_cuda_bytes=torch.cuda.max_memory_allocated(), model=MODEL_ID, model_revision=MODEL_REVISION, fp8_experts=os.environ.get("FISH_DRAMA3_FP8_EXPERTS"), cuda_graph=os.environ.get("FISH_DRAMA3_CUDA_GRAPH"))
            output.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps(receipt), flush=True)
    except Exception as exc:
        print(json.dumps(dict(request_id=rid, failed=True, error=str(exc))), flush=True)
        messages.put(exc)
    finally:
        # References never survive the request, even on an inference error.
        if ENGINE is not None:
            with LOCK:
                ENGINE.ref_by_hash.clear()
        messages.put(None)
        SLOTS.release()


@asynccontextmanager
async def lifespan(app):
    global ENGINE, DEFAULT_REFERENCE
    manifest = json.loads(Path("/opt/drama/model-manifest.json").read_text())
    verify_snapshot(os.environ["DRAMA_MODEL_DIR"], manifest["files"])
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1):
        raise RuntimeError("requires GB10 SM121 CUDA; no CPU fallback")
    if os.environ.get("FISH_DRAMA3_FP8_EXPERTS") != "1":
        raise RuntimeError("this qualified profile requires FP8 experts")
    model = Path(os.environ["DRAMA_MODEL_DIR"])
    native_queue = launch_thread_safe_queue(model, "cuda:0", torch.bfloat16, compile=False)
    codec = load_codec_model(model / "codec.pth", "cuda:0", torch.bfloat16)
    ENGINE = Engine(native_queue, codec, torch.bfloat16, False, model_family="drama3")
    audio = Path("/opt/drama/default-reference.wav").read_bytes()
    verify_default_reference(audio)
    DEFAULT_REFERENCE = ServeReferenceAudio(audio=audio, text=Path("/opt/drama/default-reference.txt").read_text().strip(), speaker=0)
    # Readiness is actual decoded native GPU inference, not just loading.
    value = validate_request(dict(text="Hello world.", max_new_tokens=200))
    messages = queue.Queue()
    assert SLOTS.acquire(blocking=False)
    synthesize(value, "startup-warmup", messages)
    while (result := messages.get()) is not None:
        if isinstance(result, Exception):
            raise result
    print("DRAMA3_READY: native GPU warmup decoded successfully", flush=True)
    yield
    native_queue.put(None)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "fishaudio", "revision": MODEL_REVISION}]}


@app.get("/health")
@app.get("/v1/health")
def health():
    return dict(status="ok", model=MODEL_ID, revision=MODEL_REVISION, source_revision=SOURCE_REVISION, default_reference_sha256=DEFAULT_REFERENCE_SHA256, started_at=STARTED, pid=os.getpid(), fp8_experts=os.environ.get("FISH_DRAMA3_FP8_EXPERTS"), cuda_graph=os.environ.get("FISH_DRAMA3_CUDA_GRAPH"), torch=torch.__version__, cudnn_enabled=torch.backends.cudnn.enabled)


@app.post("/v1/tts")
async def tts(request: Request):
    try:
        validate_http_headers(request.headers.get("content-type", ""), request.headers.get("origin"), os.environ["DRAMA_BIND_IP"], int(os.environ.get("DRAMA_PORT", "8010")))
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > MAX_BODY:
            raise HTTPException(413, "request exceeds 4 MiB")
    try:
        value = validate_request(json.loads(body))
        validate_drama3_text(value["text"], "text")
        # Reject malformed/oversized WAV before submitting work or sending 200.
        for ref in value["references"]:
            validate_drama3_text(ref["text"], "reference transcript")
            ENGINE.load_audio(base64.b64decode(ref["audio"], validate=True), ENGINE.decoder_model.sample_rate)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if not SLOTS.acquire(blocking=False):
        raise HTTPException(429, "one active and two queued requests allowed")
    rid = uuid.uuid4().hex
    messages = queue.Queue()
    threading.Thread(target=synthesize, args=(value, rid, messages), daemon=True).start()
    first = await asyncio.to_thread(messages.get)
    if isinstance(first, Exception) or first is None:
        raise HTTPException(500, "native synthesis failed; inspect scoped service logs")
    headers = {"X-Request-ID": rid, "X-Model-Revision": MODEL_REVISION, "Content-Disposition": "attachment; filename=drama3.wav"}
    if not value["streaming"]:
        return Response(first, media_type="audio/wav", headers=headers)

    async def stream():
        yield wav_chunk_header(ENGINE.decoder_model.sample_rate)
        yield first
        while True:
            part = await asyncio.to_thread(messages.get)
            if part is None:
                break
            if isinstance(part, Exception):
                # Fail by breaking transport, never return a plausible partial pass.
                raise RuntimeError("stream synthesis failed")
            yield part
    return StreamingResponse(stream(), media_type="audio/wav", headers=headers)


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """<!doctype html><html lang=en><meta charset=utf-8><title>Drama 3 GB10</title><h1>Drama 3 on GB10</h1><p>Built with Fish Audio. Personal evaluation only. Trusted LAN, no authentication. Synthetic speech; do not impersonate anyone.</p><textarea id=text rows=6 cols=70>Hello, this is Drama three running on a single Spark.</textarea><br><button id=go>Generate WAV</button><p id=status></p><audio id=audio controls></audio><p><a id=download hidden download=drama3.wav>Download WAV</a></p><script>const go=document.getElementById('go'),text=document.getElementById('text'),status=document.getElementById('status'),audio=document.getElementById('audio'),download=document.getElementById('download');let current;go.onclick=async()=>{go.disabled=true;status.textContent='Generating (serialized GPU requests)...';try{const r=await fetch('/v1/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text.value})});if(!r.ok)throw Error(await r.text());if(current)URL.revokeObjectURL(current);current=URL.createObjectURL(await r.blob());audio.src=current;download.href=current;download.hidden=false;status.textContent='Synthetic audio ready';}catch(e){status.textContent=e.message;}finally{go.disabled=false;}};</script></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--served-model-name", choices=[MODEL_ID], default=MODEL_ID)
    args = parser.parse_args()
    host, port = validate_binding(args.host, args.port)
    uvicorn.run(app, host=host, port=port, workers=1, access_log=False, limit_concurrency=16)
