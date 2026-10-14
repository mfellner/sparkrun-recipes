#!/usr/bin/env python3
"""Fail-closed, mechanically recomputed live evidence. SPDX-License-Identifier: MIT."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
import urllib.parse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fixtures import TEXTS, request_for

REVISION = "dd64a85cd82e9521f7202610ede97b367fc83946"
REQUIRED_NAMES = ["english-1", "english-2", "german-1", "german-2", "expression", "streaming", "longer", "overlap-a", "overlap-b", "reference", "post-restart"]
REQUIRED_FILES = {name + suffix for name in REQUIRED_NAMES for suffix in (".json", ".wav")} | {"asr.json", "runtime-before.json", "runtime-after.json", "platform.json", "model-verification.json", "startup.log", "startup-hash.json", "kernel.txt", "kernel-repeat.txt", "kernel-query.json", "telemetry.json", "http-negative.json", "reference.wav", "reference.txt", "reference-provenance.json", "launch-state.json", "public-image.json", "server-receipts.json", "streaming-native.wav", "job-state.yaml"}
REQUIRED_FILES |= {"reference-input.wav", "restart.json", "restart-before-runtime.json", "restart-before-request.json", "cache-before-restart.json", "cache-after-restart.json"}
REQUIRED_FILES |= {"summary.json", "protected-head.json", "browser-demo.wav", "browser-demo.json", "browser-ui.png"}
FATAL = re.compile(r"NVRM.*Xid|Out of memory: Killed|oom-kill|NV_ERR_NO_MEMORY|CUDA error|illegal memory access|NCCL ERROR|Traceback \(most recent", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_secrets(value, depth=0):
    require(depth <= 30, "evidence nesting limit")
    if isinstance(value, dict):
        for key, item in value.items():
            require(not re.search(r"authorization|password|passwd|api[_-]?key|(?:access|refresh|hf|gh|github|oauth)[_-]?token|client[_-]?secret|private[_-]?key|cookie|credential", str(key), re.I), "sensitive evidence field")
            reject_secrets(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            reject_secrets(item, depth + 1)
    elif isinstance(value, str):
        require(not re.search(r"Bearer\s+|gh[pousr]_[A-Za-z0-9]|github_pat_|hf_[A-Za-z0-9]{20,}|-----BEGIN.*PRIVATE KEY|(?:https?:)?//[^\s/]+:[^\s/@]+@", value, re.I), "secret-like evidence value")
        require(not re.search(r"\b(?:authorization|cookie|api[-_]?key|password)\s*[:=]", value, re.I), "credential-like plaintext field")
        decoded = urllib.parse.unquote(value, errors="strict")
        if decoded != value:
            reject_secrets(decoded, depth + 1)
        try:
            inner = json.loads(value)
        except (ValueError, TypeError):
            return
        if isinstance(inner, (dict, list)):
            reject_secrets(inner, depth + 1)


def validate_receipt(row, name):
    reject_secrets(row)
    require(row.get("name") == name and row.get("status") == 200 and row.get("route") == "/v1/tts", "wrong receipt identity/status/route")
    require("error" not in row, "error object in successful receipt")
    require(row.get("endpoint") == "http://192.168.178.70:8010" and row.get("model") == "fishaudio/drama-3" and row.get("model_revision") == REVISION, "wrong endpoint/model pin")
    require(re.fullmatch(r"[0-9a-f]{32}", row.get("request_id", "")) is not None, "invalid request id")
    request = request_for(name)
    if name == "reference":
        request["references"] = row["request"]["references"]
        require(len(request["references"]) == 1 and request["references"][0].get("consent") is True, "reference rights not affirmed")
    require(row.get("request") == request, "canonical request changed")
    headers = row.get("headers", {})
    require(headers.get("content-type") == "audio/wav" and headers.get("x-model-revision") == REVISION and headers.get("x-request-id") == row["request_id"], "response headers invalid")
    require(row.get("audio_file") == name + ".wav", "wrong audio file")
    start, end, wall = row.get("started_at", 0), row.get("finished_at", 0), row.get("wall_s", 0)
    require(1791928800 <= start < end and 0 < wall <= 1800 and abs((end-start)-wall) < 0.5, "bad request timing")
    audio = row.get("audio", {})
    require(audio.get("finite") is True and audio.get("channels") == 1 and audio.get("sample_rate") == 44100 and 0.5 <= audio.get("duration_s", 0) <= 180 and audio.get("rms", 0) >= 0.001, "invalid audio measurements")
    require(math.isclose(row.get("rtf", -1), wall/audio["duration_s"], rel_tol=1e-9), "RTF not recomputed")
    require(0 < row.get("first_audio_s", 0) <= wall, "TTFA invalid")
    return row


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    import yaml
    recipe = yaml.safe_load((HERE.parents[1] / "recipes/fish-audio-drama3-dgx-spark.yaml").read_text())
    require(manifest["image"] == recipe["container"], "evidence image differs from reviewed recipe")
    require(REQUIRED_FILES <= set(manifest["files"]), "required evidence missing from manifest")
    for name, expected in manifest["files"].items():
        require(Path(name).name == name, "unsafe manifest path")
        file = directory / name
        require(file.is_file() and file.stat().st_size > 0 and hashlib.sha256(file.read_bytes()).hexdigest() == expected, "evidence missing/empty/hash mismatch: " + name)
        if name.endswith(".json"):
            reject_secrets(json.loads(file.read_text()))
    # Audio facts are recomputed, not taken on faith from an updated manifest.
    from acceptance import inspect_audio
    rows = []
    for name in REQUIRED_NAMES:
        row = validate_receipt(json.loads((directory / (name + ".json")).read_text()), name)
        raw = (directory / row["audio_file"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == row["audio_sha256"], "audio digest mismatch")
        actual = inspect_audio(raw)
        require(sum(chunk["bytes"] for chunk in row["chunks"]) == len(raw) and all(0 < chunk["elapsed_s"] <= row["wall_s"] for chunk in row["chunks"]), "HTTP chunk trace does not match audio")
        for key, value in actual.items():
            require(row["audio"][key] == value, "audio result mutation: " + key)
        rows.append(row)
    a, b = [next(row for row in rows if row["name"] == name) for name in ("overlap-a", "overlap-b")]
    require(max(a["started_at"], b["started_at"]) < min(a["finished_at"], b["finished_at"]), "requests did not overlap")
    require(a["audio_sha256"] != b["audio_sha256"], "overlapping responses duplicated")
    streamed = next(row for row in rows if row["name"] == "streaming")
    require(streamed["first_audio_s"] < streamed["wall_s"] - 1, "no empirically incremental audio delivery")
    servers = json.loads((directory / "server-receipts.json").read_text())
    require(set(servers) == {row["request_id"] for row in rows}, "server/client request IDs differ")
    for row in rows:
        server = servers[row["request_id"]]
        require(server["request_id"] == row["request_id"] and server["text_sha256"] == hashlib.sha256(row["request"]["text"].encode()).hexdigest(), "server request changed")
        require(server["model"] == row["model"] and server["model_revision"] == REVISION and server["fp8_experts"] == "1" and server["cuda_graph"] == "1", "server model/mode mismatch")
        require(abs(server["queued_at"] - row["started_at"]) < 0.5 and abs(server["finished_at"] - row["finished_at"]) < 0.5 and server["queued_at"] <= server["gpu_started_at"] < server["finished_at"], "server/client timestamps do not join")
        if row["name"] != "streaming":
            require(server["audio_sha256"] == row["audio_sha256"], "server/client WAV mismatch")
        else:
            import numpy as np
            import soundfile as sf
            native = directory / "streaming-native.wav"
            require(hashlib.sha256(native.read_bytes()).hexdigest() == server["audio_sha256"], "stream native WAV mismatch")
            left, sr = sf.read(native)
            right, other_sr = sf.read(directory / "streaming.wav")
            require(sr == other_sr == 44100 and left.shape == right.shape and float(np.max(np.abs(left-right))) <= 1/32768, "stream PCM differs from native output")
    ga, gb = [servers[row["request_id"]] for row in (a, b)]
    require(ga["finished_at"] <= gb["gpu_started_at"] or gb["finished_at"] <= ga["gpu_started_at"], "GPU requests not serialized")
    asr = json.loads((directory / "asr.json").read_text())
    require(asr["model"] == "Systran/faster-whisper-small" and asr["revision"] == "536b0662742c02347bc0e980a01041f333bce120", "ASR pin mismatch")
    from asr import word_error_rate, tail_matches
    for row in rows:
        matched = [item for item in asr["rows"] if item["name"] == row["name"]]
        require(len(matched) == 1, "missing/duplicate ASR row")
        item = matched[0]
        require(item["audio_sha256"] == row["audio_sha256"] and item["expected"] == row["request"]["text"] and item["passed"] is True and word_error_rate(item["expected"], item["text"]) == item["wer"] <= 0.4, "ASR consistency failed")
        require(item["tail_matches"] is True and tail_matches(item["expected"], item["text"]), "ASR suggests truncated ending")
    reference = json.loads((directory / "reference-provenance.json").read_text())
    ref_hash = hashlib.sha256((directory / "reference-input.wav").read_bytes()).hexdigest()
    require(ref_hash == hashlib.sha256((HERE.parents[1] / "mods/drama3-gb10/default-reference.wav").read_bytes()).hexdigest() and reference == json.loads((HERE.parents[1] / "mods/drama3-gb10/reference-provenance.json").read_text()), "public reference substituted")
    require(ref_hash == reference["sha256"] and (directory / "reference.txt").read_text().strip() == reference["transcript"], "reference fixture mismatch")
    ref = next(row for row in rows if row["name"] == "reference")["request"]["references"][0]
    require(ref["audio"] == "sha256:" + ref_hash and ref["text"] == reference["transcript"], "reference request changed")
    before, after = [json.loads((directory / name).read_text()) for name in ("runtime-before.json", "runtime-after.json")]
    # Isolate this helper; other repository evidence has a different
    # capture_runtime module and must not receive ours through sys.modules.
    spec = importlib.util.spec_from_file_location("drama3_capture_runtime", HERE / "capture_runtime.py")
    if spec is None or spec.loader is None:
        raise ValueError("runtime capture helper missing")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    require(set(manifest["runtime_files"]) == set(capture.FILES), "runtime hash coverage missing")
    for runtime in (before, after):
        require(runtime["host"] == "192.168.178.70" and runtime["hostname"] == "dgx02", "wrong host identity")
        require(runtime["container"]["image_ref"] == manifest["image"] and runtime["container"]["running"] is True, "image/process not live")
        require(runtime["container"]["privileged"] is False and runtime["container"]["network"] == "host" and runtime["container"]["user"] == "1000:1000" and "no-new-privileges" in runtime["container"]["security_opt"], "security boundary changed")
        require(runtime["listener"]["address"] == "192.168.178.70:8010" and runtime["listener"]["pid"] == runtime["health"]["pid"] and runtime["listener"]["start_ticks"] > 0, "listener/process join failed")
        require(runtime["process_join"]["namespace_pids"][-1] == runtime["health"]["pid"] and runtime["process_join"]["host_start_ticks"] == runtime["listener"]["start_ticks"], "host/container PID mapping changed")
        require(runtime["health"]["model"] == "fishaudio/drama-3" and runtime["health"]["revision"] == REVISION and runtime["health"]["cudnn_enabled"] is True and runtime["health"]["fp8_experts"] == "1", "wrong live engine")
        require(runtime["health"]["default_reference_sha256"] == "4d2bd83ff0f7fe33491b03193abdf623f1b4cc2d8103972b33638d3607d86dc5" and runtime["health"]["cuda_graph"] == "1", "reference or graph mode changed")
        for path, expected in manifest["runtime_files"].items():
            require(runtime["runtime_files"].get(path) == expected and hashlib.sha256((HERE.parents[1] / "mods/drama3-gb10" / path).read_bytes()).hexdigest() == expected, "runtime file differs from reviewed source")
    require(before["container"]["id"] == after["container"]["id"] and before["listener"] == after["listener"], "service identity changed during acceptance")
    require(before["captured_at"] <= min(row["started_at"] for row in rows) and after["captured_at"] >= max(row["finished_at"] for row in rows), "runtime captures do not bracket suite")
    platform = json.loads((directory / "platform.json").read_text())
    require(platform["passed"] is True and platform["capability"] == [12, 1] and platform["cudnn_enabled"] is True and platform["triton"] == "3.7.1", "platform gate failed")
    require(platform["host"] == before["host"] and platform["image"] == manifest["image"], "platform gate not on final host/image")
    weights = json.loads((directory / "model-verification.json").read_text())
    expected_weights = json.loads((HERE.parents[1] / "mods/drama3-gb10/model-manifest.json").read_text())
    require(weights["revision"] == REVISION and weights["files"] == expected_weights["files"] and weights["passed"] is True, "model manifest verification failed")
    require(weights["host"] == before["host"] and weights["container_id"] == before["container"]["id"] and weights["image"] == manifest["image"] and weights["verified_at"] >= after["captured_at"], "weights not independently read on final instance")
    log = (directory / "startup.log").read_bytes()
    reject_secrets(log.decode())
    require(b"DRAMA3_READY" in log and b"block-FP8 W8A8" in log and not FATAL.search(log.decode()), "startup failure/missing mode")
    require(b"Drama 3 decode CUDA graphs: kv buckets" in log and b"finish=length" not in log, "graph capture missing or audio frame cap reached")
    require(json.loads((directory / "startup-hash.json").read_text())["sha256"] == hashlib.sha256(log).hexdigest(), "startup log independently hashed bytes differ")
    kernel = (directory / "kernel.txt").read_text()
    require(kernel == (directory / "kernel-repeat.txt").read_text() and "-- cursor: " in kernel and re.search(r"^\d+\.\d+ .*kernel:", kernel, re.M) and not FATAL.search(kernel), "kernel evidence incomplete/fatal")
    query = json.loads((directory / "kernel-query.json").read_text())
    require(query["host"] == "192.168.178.70" and query["since"] <= before["container"]["created_at"] and query["until"] >= after["captured_at"], "kernel window does not cover launch-to-acceptance")
    public = json.loads((directory / "public-image.json").read_text())
    require(public["anonymous"] is True and public["status"] == 200 and public["image"] == manifest["image"] and public["architecture"] == "arm64", "image not anonymously pullable ARM64")
    require(public["config_digest"] == before["container"]["image_id"] == after["container"]["image_id"], "published config differs from live image ID")
    launch = json.loads((directory / "launch-state.json").read_text())
    require(launch["recipe_sha256"] == hashlib.sha256((HERE.parents[1] / "recipes/fish-audio-drama3-dgx-spark.yaml").read_bytes()).hexdigest() and launch["container_id"] == before["container"]["id"] and launch["image"] == manifest["image"] and launch["persisted_before_acceptance"] is True, "launch-state provenance mismatch")
    job_raw = (directory / "job-state.yaml").read_bytes()
    job = yaml.safe_load(job_raw)
    reject_secrets(job)
    require(hashlib.sha256(job_raw).hexdigest() == launch["persisted_sha256"] and job["recipe_state"]["_raw"] == recipe == launch["status_meta"]["recipe_state"]["_raw"], "persisted raw recipe differs from reviewed source")
    require(job["cluster_id"] == launch["workload_id"] and launch["persistence_mtime"] <= launch["captured_at"] <= before["captured_at"], "launch state captured too late or relabeled")
    negatives = json.loads((directory / "http-negative.json").read_text())
    expected_negatives = {"empty text": 400, "oversized text": 400, "URL input": 400, "unbounded frames": 400, "structural token": 400, "invalid speaker": 400, "malformed JSON": 400, "oversized body": 413, "cross-origin": 403, "simple Content-Type": 403}
    require({row["case"]: row["status"] for row in negatives["rows"]} == expected_negatives and not any(row["reachable"] for row in negatives["unintended_interfaces"]), "bad input or unintended-interface gate failed")
    require({row["host"] for row in negatives["unintended_interfaces"]} == {"192.168.3.183", "192.168.2.183", "target loopback 127.0.0.1"} and all(row["port"] == 8010 for row in negatives["unintended_interfaces"]), "interface probes missing/relabeled")
    previous = json.loads((directory / "restart-before-runtime.json").read_text())
    witness = json.loads((directory / "restart-before-request.json").read_text())
    cb, ca = [json.loads((directory / name).read_text()) for name in ("cache-before-restart.json", "cache-after-restart.json")]
    restart = json.loads((directory / "restart.json").read_text())
    require(previous["container"]["image_id"] == before["container"]["image_id"] and previous["container"]["id"] != before["container"]["id"] and previous["listener"]["start_ticks"] != before["listener"]["start_ticks"], "no same-image process restart")
    require(restart["passed"] is True and restart["previous_container"] == previous["container"]["id"] and restart["final_container"] == before["container"]["id"], "restart receipt not joined")
    require(cb["compiler_files"] == ca["compiler_files"] and len(ca["compiler_files"]) > 0 and ca["cache_file_count"] >= cb["cache_file_count"] and ca["output_sha256"] == cb["output_sha256"] == witness["audio_sha256"] and cb["request_id"] == ca["request_id"] == witness["request_id"], "compiler/output persistence not demonstrated")
    telemetry = json.loads((directory / "telemetry.json").read_text())
    require(telemetry["host"] == "192.168.178.70" and len(telemetry["samples"]) > 5 and any(sample["gpu_util_pct"] > 0 and min(row["started_at"] for row in rows) <= sample["timestamp"] <= max(row["finished_at"] for row in rows) for sample in telemetry["samples"]), "GPU telemetry not joined to inference")
    summary = json.loads((directory / "summary.json").read_text())
    serial = [row for row in rows if not row["name"].startswith("overlap-")]
    require(summary["requests"] == len(rows) and summary["max_wer"] == max(item["wer"] for item in asr["rows"]) and summary["median_serial_rtf"] == statistics.median(row["rtf"] for row in serial) and summary["serial_rtf_range"] == [min(row["rtf"] for row in serial), max(row["rtf"] for row in serial)], "reported speech metrics changed")
    require(summary["streaming_first_pcm_s"] == streamed["first_audio_s"] and summary["streaming_wall_s"] == streamed["wall_s"] and summary["streaming_audio_duration_s"] == streamed["audio"]["duration_s"], "reported streaming metrics changed")
    require(summary["minimum_host_mem_available_bytes"] == min(sample["MemAvailable"] for sample in telemetry["samples"]) and summary["peak_cuda_allocated_bytes"] == max(server["peak_cuda_bytes"] for server in servers.values()), "reported resource metrics changed")
    browser = json.loads((directory / "browser-demo.json").read_text())
    browser_raw = (directory / "browser-demo.wav").read_bytes()
    browser_audio = inspect_audio(browser_raw)
    require(browser["url"] == "http://192.168.178.70:8010/ui" and browser["ready_state"] == 4 and browser["content_type"] == "audio/wav" and browser["bytes"] == len(browser_raw) and abs(browser["duration"] - browser_audio["duration_s"]) < 1e-6, "browser decode evidence invalid")
    protected = json.loads((directory / "protected-head.json").read_text())
    from datetime import datetime
    require(protected["sparkDash"]["id"].startswith("e55f9b6d2220") and protected["sparkDash"]["running"] is True and datetime.fromisoformat(protected["sparkDash"]["started_at"].replace("Z", "+00:00")).timestamp() < protected["baseline_epoch"] and protected["gateway"]["started_epoch"] < protected["baseline_epoch"], "protected head service restarted")
    return {"passed": True, "requests": len(rows), "image": manifest["image"], "model_revision": REVISION, "rtf_range": [min(row["rtf"] for row in rows), max(row["rtf"] for row in rows)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.directory)))
    except Exception as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        raise SystemExit(1)
