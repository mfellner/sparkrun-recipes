#!/usr/bin/env python3
"""Fail-closed, offline verifier for the exact GLM-5.3 refresh receipt."""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RECIPE = Path(os.environ.get("GLM53_RECIPE", ROOT.parents[1] / "recipes/glm-5.3-flash-exl3-dflash2-dual-spark-1m.yaml"))
CLUSTER = "sparkrun_9000d029f34ee753_fe02508ba745"
MODEL = "GLM-5.3-Flash-EXL3"
SOURCE_REV = "c190db1ae17ba8dff20129ed1f308d10c63cf37d"
MAIN_OLD_REV = "25a44fdbf16862a46b7cc9921142c6c81350af2f"
MAIN_REV = "024db9f7e9871e8efdf21538ba55af7442be3cd5"
DRAFT_OLD_REV = "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"
DRAFT_REV = "bf582e4eacc1810f76656d1811693ff6c6737d2a"
IMAGE_DIGEST = "sha256:4f30fba4248ed7d78dddda37d9ffc0fea5cc8c567c5dd98042ad808efdde9791"
IMAGE_ID = "sha256:16d9b26d541c42ba3a7a02d51f738414066026dbc87f9e62d259c25c6d2ced6a"
NCCL_VERSION = "2.30.7"
NCCL_SHA256 = "fc7ea66334edbc934aa25959b9907dbb2b91a1d2485beff18839afc45cbc08d0"
VISION_SHA256 = "8b2fc0401dba5a125ac114d6e98c610fe10128a0d0a405ef8744931eea6c0e86"
VISION_SIZE = 716
VIDEO_SHA256 = "b89ebcdbedac896cc0ee9645f92000a1780d2095cfe76aa7a0b2cd38ef93b5f0"
E2_EXTENSION_SHA256 = "673a1bbe5d4f89acdb0ad2be26f5fc0dd867d5a4e1e56d6507e9169ec231cf4c"
CHAT_TEMPLATE_SHA256 = "3050de3f995e0ae8a612f411c01621d65822ddd10f7ef6e0fb7c23bdb4a18523"
METRIC_RX = re.compile(r"^vllm:prefix_cache_hits_total(?:\{[^}]*\})?\s+([\d.eE+-]+)$")
HEX64_RX = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_ARTIFACTS = {
    "README.md", "acceptance-latest-features.json", "acceptance-xgrammar.json", "acceptance.json",
    "acceptance.py", "acceptance_latest_features.py", "acceptance_xgrammar.py", "capture_image_public.py", "capture_runtime_evidence.py",
    "generate_hf_pin_maps.py", "rollback-recipe-32db610.yaml", "test_verify_negative_controls.py",
    "verification-summary.json", "verify.py", "video-red-blue.gif", "vision-quadrants.png",
    "raw/direct-models.json", "raw/dry-run-output.json", "raw/dry-run.txt", "raw/final-pins.txt",
    "raw/hf-pins.json", "raw/image-tag-inspect.txt", "raw/image-public.json", "raw/launch-output.json", "raw/launch.log", "raw/proxy-models.json",
    "raw/runtime-success.json", "raw/upstream-head.txt",
}


class VerificationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load(name: str) -> Any:
    return json.loads((ROOT / name).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def exact_request(marker: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Reply with exactly {marker} and nothing else."}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def completion_request(messages: list[dict[str, Any]], max_tokens: int = 64, thinking: bool = False, **extra: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    request.update(extra)
    return request


def xgrammar_request(case: int, thinking: bool) -> dict[str, Any]:
    return completion_request(
        [{"role": "user", "content": f"Case {case}: think briefly, then return the required JSON object with answer 42 and case {case}."}],
        max_tokens=1024,
        thinking=thinking,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": f"case_{case}",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "integer", "const": 42},
                        "case": {"type": "integer", "const": case},
                    },
                    "required": ["answer", "case"],
                    "additionalProperties": False,
                },
            },
        },
    )


def response_message(row: dict[str, Any]) -> dict[str, Any]:
    return row["response"]["choices"][0]["message"]


def response_content(row: dict[str, Any]) -> str:
    return (response_message(row).get("content") or "").strip()


def verify_interval(row: dict[str, Any], label: str) -> None:
    require(isinstance(row.get("started_at"), (int, float)), f"{label}: missing started_at")
    require(isinstance(row.get("completed_at"), (int, float)), f"{label}: missing completed_at")
    require(row["completed_at"] >= row["started_at"], f"{label}: invalid time interval")


def verify_request(row: dict[str, Any], label: str, http_required: bool = True) -> None:
    if http_required:
        require(row.get("http", 200 if row.get("ok") else None) == 200, f"{label}: HTTP status is not 200")
    require(isinstance(row.get("request"), dict), f"{label}: request missing")
    require(row.get("request_sha256") == payload_hash(row["request"]), f"{label}: request hash mismatch")
    verify_interval(row, label)


def verify_manifest() -> None:
    manifest = ROOT / "SHA256SUMS"
    listed: dict[str, str] = {}
    for number, line in enumerate(manifest.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise VerificationError(f"SHA256SUMS:{number}: malformed line")
        digest, rel = match.groups()
        require(rel not in listed, f"SHA256SUMS: duplicate path {rel}")
        path = Path(rel)
        require(not path.is_absolute() and ".." not in path.parts, f"SHA256SUMS: unsafe path {rel}")
        listed[rel] = digest
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and "__pycache__" not in path.parts
    }
    require(REQUIRED_ARTIFACTS <= actual, f"required evidence missing: {sorted(REQUIRED_ARTIFACTS-actual)}")
    require(set(listed) == actual, f"SHA256SUMS coverage mismatch: missing={sorted(actual-set(listed))}, stale={sorted(set(listed)-actual)}")
    for rel, expected in listed.items():
        require(sha256(ROOT / rel) == expected, f"SHA256SUMS digest mismatch: {rel}")


def verify_basic(basic: dict[str, Any], recipe_sha: str) -> dict[str, Any]:
    require(basic.get("schema") == 1, "acceptance: schema")
    require(basic.get("cluster_id") == CLUSTER, "acceptance: cluster")
    require(basic.get("recipe_sha256") == recipe_sha, "acceptance: recipe hash")
    require(basic.get("model") == MODEL and basic.get("process_role") == "exact_final", "acceptance: model/process role")
    require(basic.get("acceptance_script_sha256") == sha256(ROOT / "acceptance.py"), "acceptance: script hash")
    checks = basic.get("checks", {})
    expected = {
        "direct_models", "direct_exact", "direct_concurrency_c4", "direct_long_context",
        "direct_tool_call", "direct_vision", "proxy_models", "proxy_exact", "proxy_vision",
    }
    require(set(checks) == expected, "acceptance: unexpected check set")

    for key in ("direct_models", "proxy_models"):
        row = checks[key]
        verify_interval(row, key)
        require(row.get("http") == 200, f"{key}: HTTP")
        ids = [entry.get("id") for entry in row["response"].get("data", [])]
        require(MODEL in ids, f"{key}: model missing")
    direct_model = next(entry for entry in checks["direct_models"]["response"]["data"] if entry.get("id") == MODEL)
    require(direct_model.get("root", "").endswith("/" + MAIN_REV), "direct_models: wrong snapshot")
    require(direct_model.get("max_model_len") == 1_000_000, "direct_models: wrong context")

    exact_expectations = {"direct_exact": "GLM53_DIRECT_OK", "proxy_exact": "GLM53_PROXY_OK"}
    for key, marker in exact_expectations.items():
        row = checks[key]
        verify_request(row, key)
        require(row["request"] == exact_request(marker), f"{key}: canonical request controls")
        require(response_content(row) == marker, f"{key}: semantic mismatch")
        require(row["response"]["choices"][0].get("finish_reason") == "stop", f"{key}: finish reason")

    c4 = checks["direct_concurrency_c4"]
    markers = [f"GLM53_C4_{index}_OK" for index in range(4)]
    require(c4.get("markers") == markers and len(c4.get("results", [])) == 4, "C4: shape")
    for index, (row, marker) in enumerate(zip(c4["results"], markers)):
        verify_request(row, f"C4[{index}]")
        require(row["request"] == exact_request(marker), f"C4[{index}]: canonical request controls")
        require(response_content(row) == marker, f"C4[{index}]: semantic mismatch")
    require(max(row["started_at"] for row in c4["results"]) < min(row["completed_at"] for row in c4["results"]), "C4: requests did not overlap")

    long_row = checks["direct_long_context"]
    verify_request(long_row, "long_context")
    needle = "NEEDLE_GL53_842917"
    require(long_row.get("needle") == needle and long_row.get("filler_token") == "alpha ", "long_context: recipe")
    repeats = long_row.get("filler_repeats_each_side")
    require(repeats == 55_000, "long_context: repeat count")
    filler = long_row["filler_token"] * repeats
    prompt = filler + f"\nHidden retrieval code: {needle}\n" + filler + f"\nReply with exactly {needle}."
    expected_long_request = completion_request([{"role": "user", "content": prompt}], max_tokens=32, thinking=False)
    require(long_row["request"] == expected_long_request, "long_context: canonical request controls")
    require(long_row.get("prompt_chars") == len(prompt), "long_context: prompt length")
    require(long_row.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest(), "long_context: prompt hash")
    require(response_content(long_row) == needle, "long_context: needle recovery")
    long_usage = long_row["response"]["usage"]
    require(long_usage.get("prompt_tokens") == 110_042, "long_context: prompt token count")

    tool = checks["direct_tool_call"]
    verify_request(tool, "tool")
    expected_tool_request = completion_request(
        [{"role": "user", "content": "Use the weather tool for Paris, France."}],
        max_tokens=128,
        thinking=False,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        tool_choice="auto",
    )
    require(tool["request"] == expected_tool_request, "tool: canonical request controls")
    choice = tool["response"]["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    require(choice.get("finish_reason") == "tool_calls" and len(calls) >= 1, "tool: no tool call")
    parsed = [
        (call.get("function", {}).get("name"), json.loads(call.get("function", {}).get("arguments", "{}")))
        for call in calls
    ]
    require(("get_weather", {"city": "Paris, France"}) in parsed, "tool: wrong function or arguments")

    fixture = ROOT / basic["fixture"]["path"]
    require(fixture.is_file(), "vision fixture missing")
    require(basic["fixture"]["path"] == "vision-quadrants.png", "vision fixture path")
    require(basic["fixture"]["sha256"] == VISION_SHA256 and basic["fixture"]["size"] == VISION_SIZE, "vision fixture constants")
    require(sha256(fixture) == VISION_SHA256 and fixture.stat().st_size == VISION_SIZE, "vision fixture mismatch")
    for key in ("direct_vision", "proxy_vision"):
        row = checks[key]
        verify_request(row, key)
        url = row["request"]["messages"][0]["content"][1]["image_url"]["url"]
        prefix = "data:image/png;base64,"
        require(url.startswith(prefix), f"{key}: fixture URL")
        decoded = base64.b64decode(url[len(prefix):], validate=True)
        require(hashlib.sha256(decoded).hexdigest() == VISION_SHA256, f"{key}: fixture digest")
        expected_content = [
            {"type": "text", "text": "What color is the top-left quadrant? Reply with exactly RED."},
            {"type": "image_url", "image_url": {"url": url}},
        ]
        require(row["request"] == completion_request([{"role": "user", "content": expected_content}], max_tokens=32, thinking=False), f"{key}: canonical request controls")
        require(response_content(row) == "RED", f"{key}: semantic mismatch")

    computed = {name: True for name in sorted(expected)}
    require(all(checks[name].get("passed") is True for name in expected), "acceptance: producer flags disagree")
    require(basic.get("passed") is True, "acceptance: top-level flag")
    return {"checks": computed, "long_prompt_tokens": long_usage["prompt_tokens"]}


def verify_xgrammar(record: dict[str, Any], recipe_sha: str) -> list[bool]:
    require(record.get("schema") == 1 and record.get("cluster_id") == CLUSTER, "xgrammar: identity")
    require(record.get("recipe_sha256") == recipe_sha and record.get("process_role") == "exact_final", "xgrammar: process")
    require(record.get("script_sha256") == sha256(ROOT / "acceptance_xgrammar.py"), "xgrammar: script hash")
    rows = record.get("reasoning_c4", [])
    require(len(rows) == 4, "xgrammar: C4 length")
    for index, row in enumerate(rows):
        verify_request(row, f"xgrammar[{index}]")
        request = row["request"]
        require(request == xgrammar_request(index, True), f"xgrammar[{index}]: canonical request controls")
        require(request["chat_template_kwargs"].get("enable_thinking") is True, f"xgrammar[{index}]: thinking disabled")
        schema = request["response_format"]["json_schema"]
        require(schema.get("strict") is True, f"xgrammar[{index}]: schema not strict")
        require(schema["schema"]["properties"]["answer"].get("const") == 42, f"xgrammar[{index}]: answer schema")
        require(schema["schema"]["properties"]["case"].get("const") == index, f"xgrammar[{index}]: case schema")
        parsed = json.loads(response_content(row))
        require(parsed == {"answer": 42, "case": index}, f"xgrammar[{index}]: response schema semantics")
        require(bool(response_message(row).get("reasoning")), f"xgrammar[{index}]: reasoning missing")
        require(row["response"]["choices"][0].get("finish_reason") == "stop", f"xgrammar[{index}]: finish")
        require(row.get("passed") is True, f"xgrammar[{index}]: producer flag")
    require(max(row["started_at"] for row in rows) < min(row["completed_at"] for row in rows), "xgrammar: C4 did not overlap")
    control = record["nonthinking_control"]
    verify_request(control, "xgrammar_control")
    require(control["request"] == xgrammar_request(9, False), "xgrammar control: canonical request controls")
    require(control["request"]["chat_template_kwargs"].get("enable_thinking") is False, "xgrammar control: thinking enabled")
    require(json.loads(response_content(control)) == {"answer": 42, "case": 9}, "xgrammar control: semantics")
    require(not response_message(control).get("reasoning"), "xgrammar control: reasoning present")
    require(control.get("passed") is True and record.get("passed") is True, "xgrammar: producer flags")
    return [True, True, True, True]


def verify_metric(sample: dict[str, Any], label: str) -> float:
    require(sample.get("url") == "http://127.0.0.1:8000/metrics", f"{label}: URL")
    require(isinstance(sample.get("captured_at"), (int, float)), f"{label}: timestamp")
    match = METRIC_RX.fullmatch(sample.get("metric_line", ""))
    if match is None:
        raise VerificationError(f"{label}: raw metric line")
    value = float(match.group(1))
    stored_value = sample.get("value")
    if not isinstance(stored_value, (int, float)):
        raise VerificationError(f"{label}: stored value")
    require(math.isclose(value, float(stored_value), rel_tol=0, abs_tol=0), f"{label}: parsed value")
    return value


def verify_features(record: dict[str, Any], recipe_sha: str) -> dict[str, Any]:
    require(record.get("schema") == 1 and record.get("cluster_id") == CLUSTER, "features: identity")
    require(record.get("recipe_sha256") == recipe_sha, "features: recipe")
    require(record.get("script_sha256") == sha256(ROOT / "acceptance_latest_features.py"), "features: script hash")
    checks = record.get("checks", {})
    require(set(checks) == {"apc", "reasoning_stop", "video", "kpool_long_generation"}, "features: check set")

    apc = checks["apc"]
    cold, follow = apc["cold"], apc["follow"]
    verify_request(cold, "APC cold", http_required=False)
    verify_request(follow, "APC follow", http_required=False)
    require(cold.get("ok") is True and follow.get("ok") is True, "APC: transport")
    cold_messages = cold["request"].get("messages", [])
    require(len(cold_messages) == 1 and cold_messages[0].get("role") == "user", "APC: cold message shape")
    apc_prompt = cold_messages[0].get("content", "")
    apc_match = re.fullmatch(r"FINAL-APC-([0-9a-f]{12}) (?:alpha ){7600} Reply exactly FINAL_APC_BASE[.]", apc_prompt)
    require(apc_match is not None, "APC: canonical unique prefix")
    require(cold["request"] == completion_request([{"role": "user", "content": apc_prompt}]), "APC: canonical cold controls")
    expected_follow = completion_request([
        {"role": "user", "content": apc_prompt},
        {"role": "assistant", "content": "FINAL_APC_BASE"},
        {"role": "user", "content": "Reply exactly FINAL_APC_FOLLOW."},
    ])
    require(follow["request"] == expected_follow, "APC: canonical follow controls")
    require(bool(response_content(cold)) and response_content(follow) == "FINAL_APC_FOLLOW", "APC: semantics")
    require(follow["request"]["messages"][0] == cold["request"]["messages"][0], "APC: prefix mismatch")
    require(follow["request"]["messages"][1:] == [{"role": "assistant", "content": "FINAL_APC_BASE"}, {"role": "user", "content": "Reply exactly FINAL_APC_FOLLOW."}], "APC: follow-up shape")
    before = verify_metric(apc["metrics_before"], "APC before")
    after = verify_metric(apc["metrics_after"], "APC after")
    require(cold["completed_at"] <= apc["metrics_before"]["captured_at"] <= follow["started_at"], "APC: before counter timing")
    require(follow["completed_at"] <= apc["metrics_after"]["captured_at"], "APC: after counter timing")
    delta = after - before
    require(delta > 0 and math.isclose(delta, float(apc.get("hit_delta")), rel_tol=0, abs_tol=0), "APC: delta")

    stop = checks["reasoning_stop"]
    result = stop["result"]
    verify_request(result, "reasoning_stop", http_required=False)
    require(result.get("ok") is True, "reasoning_stop: transport")
    expected_stop = completion_request(
        [{"role": "user", "content": "In your reasoning, explicitly write the exact string Question: and continue reasoning. After reasoning, put STOP_GUARD_OK as the final non-empty line."}],
        max_tokens=2048,
        thinking=True,
        stop=["Question:"],
    )
    require(result["request"] == expected_stop, "reasoning_stop: canonical request controls")
    require(result["request"].get("stop") == ["Question:"], "reasoning_stop: client stop")
    require(result["request"]["chat_template_kwargs"].get("enable_thinking") is True, "reasoning_stop: thinking")
    choice = result["response"]["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    lines = [line.strip() for line in (message.get("content") or "").splitlines() if line.strip()]
    require("Question:" in reasoning, "reasoning_stop: stop string absent from reasoning")
    require(lines and lines[-1] == "STOP_GUARD_OK", "reasoning_stop: final answer")
    require(choice.get("finish_reason") == "stop" and choice.get("stop_reason") != "Question:", "reasoning_stop: terminated at client stop")
    require(stop.get("final_answer") == lines[-1] and stop.get("reasoning_contains_stop") is True, "reasoning_stop: summary mismatch")

    video = checks["video"]
    require(video.get("fixture_sha256") == VIDEO_SHA256, "video: fixture constant")
    require(sha256(ROOT / "video-red-blue.gif") == VIDEO_SHA256, "video: committed fixture digest")
    video_bytes = None
    for key in ("direct", "proxy"):
        row = video[key]
        verify_request(row, f"video {key}", http_required=False)
        require(row.get("ok") is True and response_content(row) == "RED THEN BLUE", f"video {key}: semantics")
        url = row["request"]["messages"][0]["content"][1]["video_url"]["url"]
        prefix = "data:image/gif;base64,"
        require(url.startswith(prefix), f"video {key}: fixture URL")
        decoded = base64.b64decode(url[len(prefix):], validate=True)
        require(hashlib.sha256(decoded).hexdigest() == VIDEO_SHA256, f"video {key}: fixture digest")
        expected_video_content = [
            {"type": "text", "text": "What are the first and last colors? Reply exactly RED THEN BLUE."},
            {"type": "video_url", "video_url": {"url": url}},
        ]
        require(row["request"] == completion_request([{"role": "user", "content": expected_video_content}]), f"video {key}: canonical request controls")
        if video_bytes is None:
            video_bytes = decoded
        else:
            require(decoded == video_bytes, "video: direct/proxy fixtures differ")

    kpool = checks["kpool_long_generation"]
    row = kpool["result"]
    verify_request(row, "kpool", http_required=False)
    require(row.get("ok") is True, "kpool: transport")
    request = row["request"]
    expected_kpool = completion_request(
        [{"role": "user", "content": "Repeat the token alpha separated by spaces until the request ends."}],
        max_tokens=2300,
        thinking=False,
        ignore_eos=True,
    )
    require(request == expected_kpool, "kpool: canonical request controls")
    usage = row["response"]["usage"]
    choice = row["response"]["choices"][0]
    require(usage.get("completion_tokens") == 2300 and kpool.get("completion_tokens") == 2300, "kpool: completion count")
    require(choice.get("finish_reason") == "length" and bool(response_content(row)), "kpool: completion outcome")
    require(all(check.get("passed") is True for check in checks.values()) and record.get("passed") is True, "features: producer flags")
    return {"apc_delta": delta, "kpool_tokens": usage["completion_tokens"]}


def map_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    if row.get("lfs_sha256"):
        return (row.get("lfs_size") or row.get("size"), "lfs_sha256", row["lfs_sha256"])
    return (row.get("size"), "blob_id", row.get("blob_id"))


def map_diff(left: dict[str, tuple[Any, ...]], right: dict[str, tuple[Any, ...]]) -> dict[str, Any]:
    left_paths, right_paths = set(left), set(right)
    return {
        "left_count": len(left), "right_count": len(right),
        "added": sorted(right_paths-left_paths), "removed": sorted(left_paths-right_paths),
        "changed": sorted(path for path in left_paths & right_paths if left[path] != right[path]),
        "equal": left == right,
    }


def verify_hf_maps(record: dict[str, Any]) -> None:
    require(record.get("schema") == 3 and record.get("source_revision") == SOURCE_REV, "HF maps: identity")
    require(record.get("generator_sha256") == sha256(ROOT / "generate_hf_pin_maps.py"), "HF maps: generator hash")
    sources = record.get("sources", {})
    expected_revisions = {
        "model_old": MAIN_OLD_REV, "model_pinned": MAIN_REV,
        "draft_old": DRAFT_OLD_REV, "draft_pinned": DRAFT_REV,
    }
    require(set(sources) == set(expected_revisions), "HF maps: source set")
    all_maps: dict[str, dict[str, tuple[Any, ...]]] = {}
    serving_maps: dict[str, dict[str, tuple[Any, ...]]] = {}
    excluded_files = set(record["non_serving_files"])
    excluded_prefixes = tuple(record["non_serving_prefixes"])
    for key, revision in expected_revisions.items():
        source = sources[key]
        require(source.get("revision") == revision, f"HF maps: {key} revision")
        rows = source.get("files", [])
        paths = [row.get("path") for row in rows]
        require(paths == sorted(paths) and len(paths) == len(set(paths)), f"HF maps: {key} paths")
        for row in rows:
            require(isinstance(row.get("size"), int) and row["size"] >= 0, f"HF maps: {key}/{row.get('path')} size")
            digest = row.get("lfs_sha256") or row.get("blob_id")
            require(isinstance(digest, str) and (HEX64_RX.fullmatch(digest) or re.fullmatch(r"[0-9a-f]{40,64}", digest)), f"HF maps: {key}/{row.get('path')} identity")
        all_maps[key] = {row["path"]: map_identity(row) for row in rows}
        serving_maps[key] = {
            row["path"]: map_identity(row) for row in rows
            if row["path"] not in excluded_files and not row["path"].startswith(excluded_prefixes)
        }
    computed = {
        "model_old_to_pinned_full": map_diff(all_maps["model_old"], all_maps["model_pinned"]),
        "model_old_to_pinned_serving": map_diff(serving_maps["model_old"], serving_maps["model_pinned"]),
        "draft_old_to_pinned_full": map_diff(all_maps["draft_old"], all_maps["draft_pinned"]),
        "draft_old_to_pinned_serving": map_diff(serving_maps["draft_old"], serving_maps["draft_pinned"]),
    }
    require(record.get("comparisons") == computed, "HF maps: comparison output mismatch")
    require(computed["model_old_to_pinned_full"]["changed"] == ["README.md"], "HF maps: unexpected main full diff")
    require(computed["model_old_to_pinned_serving"]["equal"] is True, "HF maps: main serving payload changed")
    require(computed["draft_old_to_pinned_serving"]["changed"] == ["model.safetensors"], "HF maps: draft payload diff")
    origin = record.get("origin_equivalence", {})
    require(origin.get("repo") == "brandonmusic/GLM-5.3-Flash-tr3-4bpw", "HF maps: origin repo")
    require(origin.get("revision") == "5ab363a8dcf6405955fd5f99671e01a1c9fb124b", "HF maps: origin revision")
    require(HEX64_RX.fullmatch(origin.get("manifest", {}).get("manifest_sha256", "")) is not None, "HF maps: origin manifest")
    mirror_rows = origin.get("mirror_logical_payload", [])
    origin_rows = origin.get("origin_logical_payload", [])
    require(len(mirror_rows) == len(origin_rows) == 130, "HF maps: logical payload size")
    for label, rows in (("mirror", mirror_rows), ("origin", origin_rows)):
        paths = [row.get("path") for row in rows]
        require(paths == sorted(paths) and len(paths) == len(set(paths)), f"HF maps: {label} logical paths")
        for row in rows:
            require(isinstance(row.get("size"), int) and row["size"] >= 0, f"HF maps: {label}/{row.get('path')} size")
            require(HEX64_RX.fullmatch(row.get("sha256", "")) is not None, f"HF maps: {label}/{row.get('path')} SHA")
    mirror_map = {row["path"]: (row["size"], "sha256", row["sha256"]) for row in mirror_rows}
    origin_map = {row["path"]: (row["size"], "sha256", row["sha256"]) for row in origin_rows}
    origin_comparison = map_diff(mirror_map, origin_map)
    require(origin.get("comparison") == origin_comparison and origin_comparison["equal"] is True, "HF maps: mirror/origin serving mismatch")


def verify_runtime(runtime: dict[str, Any], recipe_sha: str, latest_acceptance_completed: float) -> dict[str, Any]:
    require(runtime.get("schema") == 1 and runtime.get("cluster_id") == CLUSTER, "runtime: identity")
    require(runtime.get("capture_script_sha256") == sha256(ROOT / "capture_runtime_evidence.py"), "runtime: script hash")
    captured_at = dt.datetime.fromisoformat(runtime.get("captured_at", "")).timestamp()
    started_at = runtime.get("started_at")
    completed_at = runtime.get("completed_at")
    if not isinstance(started_at, (int, float)) or not isinstance(completed_at, (int, float)):
        raise VerificationError("runtime: capture interval")
    final_boundary = runtime.get("final_acceptance_completed_at")
    if not isinstance(final_boundary, (int, float)):
        raise VerificationError("runtime: final acceptance boundary")
    require(math.isclose(float(final_boundary), latest_acceptance_completed, rel_tol=0, abs_tol=0), "runtime: final acceptance boundary")
    require(latest_acceptance_completed <= started_at <= completed_at <= latest_acceptance_completed + 7200, "runtime: capture is not joined after final acceptance")
    require(abs(captured_at - started_at) <= 5, "runtime: human/numeric capture timestamps disagree")
    commands = runtime["commands"]
    epoch_row = commands.get("postready_ok_epoch", {})
    expected_epoch_argv = ["ssh", "192.168.178.47", "docker", "exec", "sparkrun_9000d029f34ee753_fe02508ba745_node_0", "stat", "-c", "%Y", "/tmp/glm53-postready.ok"]
    require(epoch_row.get("returncode") == 0 and epoch_row.get("argv") == expected_epoch_argv, "runtime: readiness epoch provenance")
    readiness_epoch = epoch_row.get("stdout", "").strip()
    require(readiness_epoch.isdigit(), "runtime: readiness epoch")
    require(commands["sparkrun_status"].get("returncode") == 0, "runtime: status command")
    status = json.loads(commands["sparkrun_status"]["stdout"])
    require(CLUSTER in status.get("groups", {}), "runtime: cluster missing")
    meta = status["groups"][CLUSTER]["meta"]
    require(meta["effective_container_image"].endswith(IMAGE_DIGEST), "runtime: image digest")
    raw_recipe = meta["recipe_state"]["_raw"]
    require(raw_recipe["model_revision"] == MAIN_REV, "runtime: main revision")
    require(raw_recipe["distribution_config"]["models"]["entries"][1]["revision"] == DRAFT_REV, "runtime: draft revision")
    require(raw_recipe["defaults"]["max_num_batched_tokens"] == 7168, "runtime: batched token limit")
    require(math.isclose(float(raw_recipe["defaults"]["gpu_memory_utilization"]), 0.86, rel_tol=0, abs_tol=0), "runtime: GPU memory profile")
    require(raw_recipe["env"]["GLM53_INDEXER_WORKSPACE"] == "rightsize", "runtime: indexer workspace mode")
    require(raw_recipe["env"]["GLM53_SPINWAIT_MS"] == "stock", "runtime: spinwait mode")
    require('"draft_tensor_parallel_size":2' in raw_recipe["defaults"]["dflash_speculative_config"], "runtime: draft TP")
    require(hashlib.sha256(RECIPE.read_bytes()).hexdigest() == recipe_sha, "runtime: recipe moved")

    direct_models = commands["direct_models"]
    require(direct_models.get("http") == 200, "runtime: direct models HTTP")
    models = json.loads(direct_models["body"])
    require(models["data"][0]["id"] == MODEL and models["data"][0]["root"].endswith("/" + MAIN_REV), "runtime: direct model")
    proxy_models = commands["proxy_models"]
    require(proxy_models.get("http") == 200 and MODEL in [row["id"] for row in json.loads(proxy_models["body"])["data"]], "runtime: proxy model")
    for key in ("direct_models", "proxy_models", "direct_marker", "proxy_marker"):
        require(started_at <= commands[key].get("started_at", 0) <= commands[key].get("completed_at", 0) <= completed_at, f"runtime: {key} outside capture interval")
    for key, marker, url in (
        ("direct_marker", "RAW_RUNTIME_DIRECT_OK", "http://127.0.0.1:8000/v1/chat/completions"),
        ("proxy_marker", "RAW_RUNTIME_PROXY_OK", "http://127.0.0.1:4000/v1/chat/completions"),
    ):
        row = commands[key]
        require(row.get("url") == url and row.get("marker") == marker and row.get("http") == 200, f"runtime: {key} transport")
        body = {"model": MODEL, "messages": [{"role": "user", "content": f"Reply with exactly {marker}"}], "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}
        require(row.get("request_sha256") == payload_hash(body), f"runtime: {key} request hash")
        require(response_content({"response": json.loads(row["body"])}) == marker, f"runtime: {key} semantics")

    fatal_log_patterns = (
        "Traceback (most recent call last)", "CUDA error:", "OutOfMemoryError",
        "illegal memory access", "Engine core encountered an issue", "ncclSystemError", "ncclInternalError",
    )
    fatal_kernel_patterns = ("NVRM: Xid", "oom-kill", "Out of memory: Killed process")
    host_summary = {}
    require(set(runtime.get("hosts", {})) == {"192.168.178.47", "192.168.178.46"}, "runtime: host set")
    for host, rows in runtime["hosts"].items():
        container = rows.get("container")
        require(container == ("sparkrun_9000d029f34ee753_fe02508ba745_node_0" if host.endswith(".47") else "sparkrun_9000d029f34ee753_fe02508ba745_node_1"), f"runtime: {host} container identity")
        required_commands = ("docker_inspect", "process_table", "serve_log", "kernel_log", "kernel_log_postready", "kernel_log_after_acceptance", "rdma_link", "hca_rate_state", "overlay_runtime", "nccl_runtime")
        for name in required_commands:
            require(rows[name].get("returncode") == 0, f"runtime: {host}/{name}: {rows[name].get('stderr')}")
        inspect = json.loads(rows["docker_inspect"]["stdout"])[0]
        require(rows["docker_inspect"].get("argv") == ["ssh", host, "docker", "inspect", container], f"runtime: {host} inspect provenance")
        expected_process_argv = ["ssh", host, "docker", "exec", container, "ps", "-eo", "pid,args"]
        require(rows["process_table"].get("argv") == expected_process_argv, f"runtime: {host} process-table provenance")
        require(inspect["State"].get("Running") is True and inspect["State"].get("OOMKilled") is False, f"runtime: {host} container state")
        require(inspect.get("RestartCount") == 0, f"runtime: {host} restart count")
        host_config = inspect["HostConfig"]
        require(inspect["Config"].get("User") == "root", f"runtime: {host} user")
        require(host_config.get("Privileged") is False, f"runtime: {host} privileged mode")
        require(host_config.get("NetworkMode") == "host" and host_config.get("IpcMode") == "host", f"runtime: {host} network/IPC")
        require(host_config.get("CapAdd") == ["CAP_IPC_LOCK"], f"runtime: {host} capabilities")
        require("no-new-privileges" in (host_config.get("SecurityOpt") or []), f"runtime: {host} no-new-privileges")
        require(any(device.get("PathOnHost") == "/dev/infiniband" for device in (host_config.get("Devices") or [])), f"runtime: {host} RDMA device")
        env = inspect["Config"]["Env"]
        for expected in ("ABLIT=0", "EXL3_FUSED_MOE=1", "EXL3_MOE_ROW_TILE=0"):
            require(expected in env, f"runtime: {host} env {expected}")
        logs = rows["serve_log"]["stdout"]
        expected_serve_log_argv = ["ssh", host, "docker", "exec", container, "cat", "/tmp/sparkrun_serve.log"]
        require(rows["serve_log"].get("argv") == expected_serve_log_argv, f"runtime: {host} serve-log provenance")
        require(not any(pattern in logs for pattern in fatal_log_patterns), f"runtime: {host} fatal serve log")
        require("[glm53-indexer-workspace] rightsize:" in logs and "[glm53-indexer-workspace] builder:" in logs, f"runtime: {host} rightsize workspace diagnostics")
        expected_rank = "0" if host.endswith(".47") else "1"
        diag_matches = re.findall(r"\(Worker_TP([01]) pid=(\d+)\).*exl3 e2 diag (schema=1[^\r\n]*)", logs)
        require(diag_matches, f"runtime: {host} E2 diagnostics missing")
        diag_rank, diag_pid, diag_line = diag_matches[-1]
        require(diag_rank == expected_rank, f"runtime: {host} E2 log rank")
        e2_process_pattern = rf"^\s*{re.escape(diag_pid)}\s+VLLM::Worker_TP{expected_rank}\b"
        require(re.search(e2_process_pattern, rows["process_table"]["stdout"], re.MULTILINE) is not None, f"runtime: {host} E2 PID is not the captured worker")
        diag = dict(token.split("=", 1) for token in diag_line.split() if "=" in token)
        for key, value in {
            "schema": "1", "configured_tier": "kernel", "effective_tier": "kernel",
            "tier_reason": "kernel_ok", "sym_exl3_moe": "1", "sym_fat_gemm": "1",
            "sym_fat_gemm_scatter": "1", "cap": "12.1", "cap_ok": "1",
            "tp_rank": expected_rank, "tp_size": "2",
        }.items():
            require(diag.get(key) == value, f"runtime: {host} E2 diagnostic {key}")
        require(int(diag.get("prefill_layer_calls", "0")) > 0, f"runtime: {host} E2 prefill calls")
        require(int(diag.get("direct_calls", "0")) > 0 and int(diag.get("scatter_calls", "0")) > 0, f"runtime: {host} E2 direct/scatter calls")
        kernel = rows["kernel_log"]["stdout"]
        require(not any(pattern in kernel for pattern in fatal_kernel_patterns), f"runtime: {host} fatal kernel log")
        postready_kernel = rows["kernel_log_postready"]["stdout"]
        expected_postready_argv = ["ssh", host, f"journalctl -k --since '@{readiness_epoch}' --no-pager"]
        require(rows["kernel_log_postready"].get("argv") == expected_postready_argv, f"runtime: {host} postready kernel provenance")
        require(not any(pattern in postready_kernel for pattern in fatal_kernel_patterns), f"runtime: {host} fatal postready kernel log")
        postready_allocations = postready_kernel.count("NV_ERR_NO_MEMORY")
        require(postready_allocations <= 10, f"runtime: {host} unbounded postready allocation notices")
        final_kernel = rows["kernel_log_after_acceptance"]["stdout"]
        expected_final_kernel_argv = ["ssh", host, f"journalctl -k --since '@{int(latest_acceptance_completed)}' --no-pager"]
        require(rows["kernel_log_after_acceptance"].get("argv") == expected_final_kernel_argv, f"runtime: {host} final kernel provenance")
        require("NV_ERR_NO_MEMORY" not in final_kernel, f"runtime: {host} allocation notice after final acceptance")
        require(not any(pattern in final_kernel for pattern in fatal_kernel_patterns), f"runtime: {host} fatal kernel event after acceptance")
        rate = rows["hca_rate_state"]["stdout"]
        require(rate.count("200 Gb/sec") >= 2 and rate.count("ACTIVE") >= 2, f"runtime: {host} HCA state")
        overlay = rows["overlay_runtime"]["stdout"]
        require(overlay.count("c9e765e13747cde82840c7af44945b7f06a1dee176df472dcebd1d858f9a5843") >= 2, f"runtime: {host} EXL3 overlay")
        require("b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac" in overlay, f"runtime: {host} draft payload")
        require(E2_EXTENSION_SHA256 in overlay, f"runtime: {host} E2 extension digest")
        require(CHAT_TEMPLATE_SHA256 in overlay, f"runtime: {host} chat template digest")
        marks_match = re.search(r"^indexer_workspace_marks=(\d+)$", overlay, re.MULTILINE)
        require(marks_match is not None and int(marks_match.group(1)) > 0, f"runtime: {host} indexer workspace patch")
        require("spinwait_stock=1" in overlay, f"runtime: {host} stock spinwait source")
        require("e2_symbols=exl3_moe,exl3_fat_gemm,exl3_fat_gemm_scatter" in overlay, f"runtime: {host} E2 symbols")
        for expected in ("EXL3_FAT_KERNEL=1", "EXL3_FAT_BATCHED=0", "EXL3_FAT_SORTED=0", "GLM53_INDEXER_WORKSPACE=rightsize", "GLM53_SPINWAIT_MS=stock"):
            require(expected in overlay, f"runtime: {host} E2 environment {expected}")
        require("\n1\n2\n" in overlay and "ABLIT=0" in overlay and "EXL3_MOE_ROW_TILE=0" in overlay, f"runtime: {host} patch markers")
        nccl = rows["nccl_runtime"]["stdout"]
        nccl_cmd = (
            "pid=$(pgrep -fo 'vllm serve'); test -n \"$pid\"; "
            "lib=$(awk '/libnccl[.]so[.]2/{print $6; exit}' /proc/$pid/maps); test -n \"$lib\"; "
            "printf 'pid=%s\\npath=%s\\n' \"$pid\" \"$(readlink -f \"$lib\")\"; "
            "printf 'sha256='; sha256sum \"$lib\" | cut -d' ' -f1; "
            "python3 -c \"import importlib.metadata as m; print('package_version='+m.version('nvidia-nccl-cu13'))\""
        )
        expected_nccl_argv = ["ssh", host, f"docker exec {shlex.quote(container)} bash -lc {shlex.quote(nccl_cmd)}"]
        require(rows["nccl_runtime"].get("argv") == expected_nccl_argv, f"runtime: {host} NCCL command provenance")
        nccl_values = dict(line.split("=", 1) for line in nccl.splitlines() if "=" in line)
        require(set(nccl_values) == {"pid", "path", "sha256", "package_version"}, f"runtime: {host} NCCL receipt shape")
        require(nccl_values["pid"].isdigit(), f"runtime: {host} NCCL PID")
        require(nccl_values["path"] == "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2", f"runtime: {host} loaded NCCL path")
        require(nccl_values["sha256"] == NCCL_SHA256, f"runtime: {host} loaded NCCL digest")
        require(nccl_values["package_version"] == NCCL_VERSION, f"runtime: {host} NCCL package version")
        process_pattern = rf"^\s*{re.escape(nccl_values['pid'])}\s+.*\bvllm\s+serve\b"
        require(re.search(process_pattern, rows["process_table"]["stdout"], re.MULTILINE) is not None, f"runtime: {host} NCCL PID is not the serving process")
        host_summary[host] = {"container_running": True, "hca_200g_links": rate.count("200 Gb/sec"), "nccl_version": NCCL_VERSION, "postready_allocation_notices": postready_allocations, "post_acceptance_allocation_notices": 0, "fatal_log_patterns": 0, "xid_or_oom_patterns": 0}
    head = runtime["hosts"]["192.168.178.47"]
    require(head["postready_rc"].get("returncode") == 0 and head["postready_rc"]["stdout"].strip() == "0", "runtime: postready rc")
    require(head["postready_ok"].get("returncode") == 0 and bool(head["postready_ok"]["stdout"].strip()), "runtime: postready receipt")
    require("postready gate OK" in head["postready_log"]["stdout"], "runtime: postready log")
    return host_summary


def verify_provenance_text() -> None:
    pins = dict(line.split("=", 1) for line in (ROOT / "raw/final-pins.txt").read_text().splitlines())
    require(pins == {"upstream_main": SOURCE_REV, "main_model": MAIN_REV, "draft_model": DRAFT_REV, "image_digest": IMAGE_DIGEST, "image_id": IMAGE_ID}, "final pins receipt")
    upstream = (ROOT / "raw/upstream-head.txt").read_text().splitlines()
    require(upstream and upstream[0] == SOURCE_REV, "upstream head receipt")
    image = (ROOT / "raw/image-tag-inspect.txt").read_text()
    require(re.search(rf"^Digest:\s+{re.escape(IMAGE_DIGEST)}$", image, re.MULTILINE) is not None, "image tag receipt")
    require(re.search(rf"^ImageID:\s+{re.escape(IMAGE_ID)}$", image, re.MULTILINE) is not None, "image ID receipt")
    public = load("raw/image-public.json")
    require(public.get("schema") == 2 and public.get("capture_script_sha256") == sha256(ROOT / "capture_image_public.py"), "public image capture script")
    require(public.get("expected_digest") == IMAGE_DIGEST and public.get("source_revision") == SOURCE_REV, "public image expected identity")
    anonymous = public["anonymous_registry"]
    require(anonymous.get("token_http") == 200 and anonymous.get("manifest_http") == 200, "public image anonymous access")
    require(anonymous.get("token_url") == "https://ghcr.io/token?scope=repository:mfellner/glm-5.3-flash-2x-dgx-sparks:pull&service=ghcr.io", "public image token URL")
    require(anonymous.get("manifest_url") == "https://ghcr.io/v2/mfellner/glm-5.3-flash-2x-dgx-sparks/manifests/c190db1", "public image manifest URL")
    manifest_body = base64.b64decode(anonymous.get("manifest_body_base64", ""), validate=True)
    manifest_sha = hashlib.sha256(manifest_body).hexdigest()
    require(len(manifest_body) == anonymous.get("manifest_size") and manifest_sha == anonymous.get("manifest_body_sha256"), "public image retained manifest bytes")
    require(anonymous.get("docker_content_digest") == IMAGE_DIGEST == "sha256:" + manifest_sha, "public image registry digest")
    manifest_json = json.loads(manifest_body)
    require(manifest_json.get("schemaVersion") == 2 and isinstance(manifest_json.get("layers"), list) and len(manifest_json["layers"]) > 0, "public image manifest semantics")
    expected_local_argv = ["docker", "image", "inspect", f"ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks@{IMAGE_DIGEST}"]
    expected_worker_argv = ["ssh", "192.168.178.46", f"docker image inspect ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks@{IMAGE_DIGEST}"]
    require(public["local_inspect"].get("argv") == expected_local_argv and public["worker_inspect"].get("argv") == expected_worker_argv, "public image inspect commands")
    for label in ("local_inspect", "worker_inspect"):
        inspect = json.loads(public[label]["stdout"])[0]
        require(inspect.get("Id") == IMAGE_ID and f"ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks@{IMAGE_DIGEST}" in inspect.get("RepoDigests", []), f"public image {label} identity")
        require(inspect["Config"]["Labels"].get("glm53.recipe.stamp") == SOURCE_REV, f"public image {label} source label")
    package = json.loads(public["package_api"]["stdout"])
    require(public["package_api"].get("argv") == ["gh", "api", "/user/packages/container/glm-5.3-flash-2x-dgx-sparks"], "package API command")
    require(package.get("name") == "glm-5.3-flash-2x-dgx-sparks" and package.get("visibility") == "public", "public package visibility")
    launch_record = load("raw/launch-output.json")
    launch = launch_record.get("stdout", "")
    require(launch_record.get("schema") == 1 and isinstance(launch, str), "launch receipt schema")
    require(launch_record.get("captured_command") == "sparkrun run recipes/glm-5.3-flash-exl3-dflash2-dual-spark-1m.yaml --cluster vacation-pair2 --no-follow", "launch captured command")
    require(hashlib.sha256(launch.encode()).hexdigest() == launch_record.get("stdout_sha256"), "launch stdout hash")
    normalized_launch = "\n".join(line.rstrip() for line in launch.splitlines()) + "\n"
    require((ROOT / "raw/launch.log").read_text() == normalized_launch, "launch normalized text")
    for marker in (
        "sparkrun v0.3.6",
        CLUSTER,
        IMAGE_DIGEST,
        MAIN_REV,
        DRAFT_REV,
    ):
        require(marker in launch, f"launch receipt missing: {marker}")
    dry_record = load("raw/dry-run-output.json")
    require(dry_record.get("schema") == 1, "dry-run receipt schema")
    require(dry_record.get("captured_command") == "sparkrun run recipes/glm-5.3-flash-exl3-dflash2-dual-spark-1m.yaml --cluster vacation-pair2 --dry-run", "dry-run captured command")
    dry_run = dry_record.get("stdout", "")
    require(isinstance(dry_run, str) and hashlib.sha256(dry_run.encode()).hexdigest() == dry_record.get("stdout_sha256"), "dry-run stdout hash")
    require(dry_run.endswith("\n\n") and (ROOT / "raw/dry-run.txt").read_text() == dry_run[:-1], "dry-run normalized text")
    for marker in (
        f"ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks@{IMAGE_DIGEST}",
        "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw",
        "--gpu-memory-utilization 0.86",
        "--max-model-len 1000000",
        "--max-num-batched-tokens 7168",
        f"snapshots/{MAIN_REV}",
        f"snapshots/{DRAFT_REV}",
        '\"draft_tensor_parallel_size\":2',
    ):
        require(marker in dry_run, f"dry-run receipt missing: {marker}")
    context_match = re.search(r"Max context tokens:\s*([0-9,]+)", dry_run)
    require(context_match is not None and int(context_match.group(1).replace(",", "")) > 1_000_000, "dry-run context fit")


def main() -> int:
    verify_manifest()
    recipe_sha = sha256(RECIPE)
    basic = load("acceptance.json")
    xgrammar = load("acceptance-xgrammar.json")
    features = load("acceptance-latest-features.json")
    runtime = load("raw/runtime-success.json")
    hf_maps = load("raw/hf-pins.json")
    for name, record in (("basic", basic), ("xgrammar", xgrammar), ("features", features), ("runtime", runtime)):
        require(record.get("cluster_id") == CLUSTER, f"{name}: cluster join")
    for name, record in (("basic", basic), ("xgrammar", xgrammar), ("features", features)):
        require(record.get("recipe_sha256") == recipe_sha, f"{name}: recipe join")
    basic_result = verify_basic(basic, recipe_sha)
    xgrammar_result = verify_xgrammar(xgrammar, recipe_sha)
    feature_result = verify_features(features, recipe_sha)
    verify_hf_maps(hf_maps)
    latest_acceptance_completed = max(
        float(basic["completed_at"]), float(xgrammar["completed_at"]), float(features["completed_at"])
    )
    host_summary = verify_runtime(runtime, recipe_sha, latest_acceptance_completed)
    verify_provenance_text()
    summary = {
        "schema": 2,
        "passed": True,
        "verification_mode": "semantics-recomputed",
        "cluster_id": CLUSTER,
        "recipe_sha256": recipe_sha,
        "source_revision": SOURCE_REV,
        "main_revision": MAIN_REV,
        "draft_revision": DRAFT_REV,
        "image_digest": IMAGE_DIGEST,
        "main_serving_payload_equal_to_prior_pin": True,
        "basic_checks": basic_result["checks"],
        "long_context_prompt_tokens": basic_result["long_prompt_tokens"],
        "xgrammar_reasoning_c4": xgrammar_result,
        "xgrammar_nonthinking_control": True,
        "apc_hit_delta": feature_result["apc_delta"],
        "reasoning_stop_passed": True,
        "video_passed_direct_and_proxy": True,
        "kpool_long_generation_tokens": feature_result["kpool_tokens"],
        "postready_gate_rc": 0,
        "hosts": host_summary,
    }
    (ROOT / "verification-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError, VerificationError, json.JSONDecodeError) as error:
        print(f"VERIFICATION FAILED: {error}")
        raise SystemExit(1)
