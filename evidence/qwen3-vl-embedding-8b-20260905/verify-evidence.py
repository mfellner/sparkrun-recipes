#!/usr/bin/env python3
"""Recompute committed claims from lossless evidence and exercise negative controls."""

import base64
import copy
import gzip
import hashlib
import json
import math
import pathlib
import re
import uuid
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MODEL = "qwen3-vl-embedding-8b"
ENDPOINT = "http://192.168.178.51:8000/v1/embeddings"
INSTRUCTION = "Retrieve images or text relevant to the user's query."
EXPECTED_NONREG = {
    "glm_direct": "GLM-ALIVE-531",
    "qwen_direct": "QWEN-ALIVE-381",
    "glm_proxy": "GLM-PROXY-531",
    "qwen_proxy": "QWEN-PROXY-381",
}
EXPECTED_RECEIPTS = {
    "host_identity",
    "container_inspect",
    "image_inspect",
    "runtime_versions",
    "model_snapshot",
    "cache_identity_and_writability",
    "container_processes",
    "gpu_processes",
    "host_memory",
    "serve_log",
    "kernel_fatal_scan",
}
EXPECTED_RECIPE_SHA = "320c53936ae2849d8fe30193cc3c3842b7ccff35029d6c7e9a1c3ebd7c8f3b26"
EXPECTED_IMAGE_ID = "sha256:afff73b9148a454dec9188f0b366c64f33bdccf5b15c57060724ba16ef666549"
EXPECTED_IMAGE_DIGEST = "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5@sha256:f92b4a1a476fd1e235e97df2a623c04c24b69d607a78a91f5084627ec7bd4266"
EXPECTED_FIXTURES = {
    "red-circle-EMBER-742.png": "d1dc4e6a7e4e5ec3c0737ffcabbd3e6a69885f01d1fcbeaec5f0038f1f67bfd5",
    "blue-square-OCEAN-319.png": "d640cd314d32a64e9bfd648d1226c85fb0d1e3b834bfd4faeb0627632e4cf71f",
    "ocr-card-EMBER-742.png": "12e25e7f5a1ff1e708eef6d93f0b0a2c510c50a0fc30a1ea0a7bd82b7a2f840b",
    "ocr-card-OCEAN-319.png": "5395afb729a0b5225f0c624592bf74fdd807bc00161e648f0d656b4c8b7020c0",
}
TEXT_BATCH = [
    "A red circle marked EMBER-742.",
    "A blue square marked OCEAN-319.",
    "A golden retriever running on a beach.",
    "A database migration using PostgreSQL.",
]
MRL_INPUT = "A compact vector dimension check."
VISUAL_QUERIES = [
    "Retrieve the image with a red circle on the left and code EMBER-742.",
    "Retrieve the image with a blue square on the right and code OCEAN-319.",
]
OCR_QUERIES = [
    "Retrieve the image displaying token EMBER-742.",
    "Retrieve the image displaying token OCEAN-319.",
]
CONCURRENT_INPUTS = [f"Concurrent embedding request {index}: cobalt archive." for index in range(4)]
BOILERPLATE_UNIT = "Telemetry nominal. Cooling stable. Power rails stable. "
BOILERPLATE_REPEATS = 1200
LONG_QUERY = "Retrieve the report containing maintenance token ORBIT-991 and a replaced coolant valve."
LONG_POSITIVE_NEEDLE = " Maintenance token ORBIT-991 confirms the coolant valve was replaced. "
LONG_NEGATIVE_NEEDLE = " Maintenance token LUNAR-224 confirms the intake filter was inspected. "
EXPECTED_REQUEST_CONTROLS = {
    "instruction": INSTRUCTION,
    "text_batch": TEXT_BATCH,
    "mrl_input": MRL_INPUT,
    "visual_queries": VISUAL_QUERIES,
    "controlled_ocr_queries": OCR_QUERIES,
    "concurrency_inputs": CONCURRENT_INPUTS,
    "boilerplate_unit": BOILERPLATE_UNIT,
    "boilerplate_repeats_per_side": BOILERPLATE_REPEATS,
    "long_query": LONG_QUERY,
    "long_positive_needle": LONG_POSITIVE_NEEDLE,
    "long_negative_needle": LONG_NEGATIVE_NEEDLE,
}
EXPECTED_COMMAND_PARTS = {
    "host_identity": ["hostname"],
    "container_inspect": ["docker inspect", "sparkrun_7d7b52c2c9174082_15b33f409193_solo"],
    "image_inspect": ["docker image inspect", "sha256:f92b4a1a476fd1e235e97df2a623c04c24b69d607a78a91f5084627ec7bd4266"],
    "runtime_versions": ["docker exec", "sparkrun_7d7b52c2c9174082_15b33f409193_solo", "torch", "vllm"],
    "model_snapshot": ["docker exec", "2c4565515e0f265c6511776e7193b22c0968ddc7", "model.safetensors.index.json"],
    "cache_identity_and_writability": ["docker exec", "test -w", "sparkrun-runtime-cache/qwen3-vl-embedding-8b/vllm"],
    "container_processes": ["docker top", "sparkrun_7d7b52c2c9174082_15b33f409193_solo"],
    "gpu_processes": ["nvidia-smi", "used_memory"],
    "host_memory": ["free -h"],
    "serve_log": ["docker exec", "/tmp/sparkrun_serve.log"],
    "kernel_fatal_scan": ["journalctl", "--since", "NVRM|Xid|oom-kill"],
}
REQUIRED_ARTIFACTS = {
    "README.md",
    "acceptance.py",
    "acceptance-result.json",
    "acceptance-raw.json.gz",
    "red-circle-EMBER-742.png",
    "blue-square-OCEAN-319.png",
    "ocr-card-EMBER-742.png",
    "ocr-card-OCEAN-319.png",
    "non-regression.py",
    "non-regression-result.json",
    "capture-runtime.py",
    "runtime-receipt.json",
    "runtime-audit.json",
    "refresh-derived.py",
    "verify-evidence.py",
    "../../recipes/qwen3-vl-embedding-8b-dgx-spark.yaml",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_ns(value):
    return int(datetime.fromisoformat(value).timestamp() * 1_000_000_000)


def l2(vector):
    return math.sqrt(sum(value * value for value in vector))


def cosine(left, right):
    return sum(a * b for a, b in zip(left, right)) / (l2(left) * l2(right))


def finite_vector(vector, dimensions=4096):
    return isinstance(vector, list) and len(vector) == dimensions and all(math.isfinite(v) for v in vector)


def equivalent(left, right, tolerance=1e-12):
    if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None:
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(equivalent(left[key], right[key], tolerance) for key in left)
    return left == right


def data_uri(filename):
    return "data:image/png;base64," + base64.b64encode((ROOT / filename).read_bytes()).decode()


def chat_payload(text="", image_uri=None):
    content = []
    if image_uri:
        content.append({"type": "image_url", "image_url": {"url": image_uri}})
    content.append({"type": "text", "text": text})
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ],
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
    }


def expected_requests():
    boilerplate = BOILERPLATE_UNIT * BOILERPLATE_REPEATS
    requests = {
        "text_batch": {"model": MODEL, "input": TEXT_BATCH, "encoding_format": "float"},
        "mrl_1024": {"model": MODEL, "input": MRL_INPUT, "dimensions": 1024},
        "cross_red_query": chat_payload(VISUAL_QUERIES[0]),
        "cross_blue_query": chat_payload(VISUAL_QUERIES[1]),
        "cross_red_image": chat_payload(image_uri=data_uri("red-circle-EMBER-742.png")),
        "cross_blue_image": chat_payload(image_uri=data_uri("blue-square-OCEAN-319.png")),
        "ocr_ember_query": chat_payload(OCR_QUERIES[0]),
        "ocr_ocean_query": chat_payload(OCR_QUERIES[1]),
        "ocr_ember_image": chat_payload(image_uri=data_uri("ocr-card-EMBER-742.png")),
        "ocr_ocean_image": chat_payload(image_uri=data_uri("ocr-card-OCEAN-319.png")),
        "long_query": chat_payload(LONG_QUERY),
        "long_positive": chat_payload(boilerplate + LONG_POSITIVE_NEEDLE + boilerplate),
        "long_negative": chat_payload(boilerplate + LONG_NEGATIVE_NEEDLE + boilerplate),
    }
    for index, text in enumerate(CONCURRENT_INPUTS):
        requests[f"concurrency_{index}"] = {"model": MODEL, "input": text}
    return requests


def receipt_map(raw):
    receipts = raw.get("receipts", [])
    if not isinstance(receipts, list):
        return {}
    return {item.get("name"): item for item in receipts if isinstance(item, dict)}


def response_vectors(receipt):
    rows = sorted(receipt["response"]["data"], key=lambda item: item["index"])
    return [row["embedding"] for row in rows]


def single_vector(receipts, name):
    vectors = response_vectors(receipts[name])
    if len(vectors) != 1:
        raise ValueError(f"{name} did not contain one vector")
    return vectors[0]


def validate_acceptance(data, raw):
    failures = []
    raw_path = ROOT / data.get("raw_artifact", "<missing>")
    try:
        uuid.UUID(data["run_id"])
        if (
            raw.get("schema") != 1
            or raw.get("run_id") != data["run_id"]
            or raw.get("started_utc") != data["timestamp_utc"]
            or raw.get("finished_utc") != data["finished_utc"]
            or raw.get("endpoint") != ENDPOINT
            or raw.get("model") != MODEL
            or data.get("raw_artifact") != "acceptance-raw.json.gz"
            or data.get("raw_bytes") != raw_path.stat().st_size
            or data.get("raw_sha256") != sha256(raw_path)
        ):
            failures.append("raw_identity")
    except (KeyError, OSError, TypeError, ValueError):
        failures.append("raw_identity_parse")

    if data.get("base_url") != "http://192.168.178.51:8000" or data.get("model") != MODEL:
        failures.append("acceptance_target")
    if data.get("request_controls") != EXPECTED_REQUEST_CONTROLS:
        failures.append("request_controls")
    if data.get("fixtures") != EXPECTED_FIXTURES:
        failures.append("fixture_manifest")
    for filename, expected_hash in EXPECTED_FIXTURES.items():
        fixture = ROOT / filename
        if not fixture.is_file() or sha256(fixture) != expected_hash:
            failures.append(f"fixture:{filename}")

    receipts = receipt_map(raw)
    expected = expected_requests()
    if set(receipts) != set(expected):
        failures.append("raw_receipt_set")
    else:
        for name, payload in expected.items():
            receipt = receipts[name]
            if receipt.get("request") != payload:
                failures.append(f"raw_request:{name}")
            if name != "mrl_1024" and (
                receipt.get("status") != 200
                or receipt.get("response", {}).get("model") != MODEL
            ):
                failures.append(f"raw_response_identity:{name}")
            if not (
                isinstance(receipt.get("started_unix_ns"), int)
                and isinstance(receipt.get("finished_unix_ns"), int)
                and receipt["started_unix_ns"] < receipt["finished_unix_ns"]
            ):
                failures.append(f"raw_interval:{name}")

    checks = data.get("checks", {})
    try:
        text_receipt = receipts["text_batch"]
        vectors = response_vectors(text_receipt)
        metrics = {
            "status": text_receipt["status"],
            "count": len(vectors),
            "dimensions": sorted({len(vector) for vector in vectors}),
            "norm_range": [min(l2(vector) for vector in vectors), max(l2(vector) for vector in vectors)],
            "usage": text_receipt["response"].get("usage", {}),
        }
        stored = checks["text_batch"]
        if not all(finite_vector(vector) for vector in vectors):
            failures.append("text_raw_vectors")
        if not all(equivalent(stored.get(key), value) for key, value in metrics.items()):
            failures.append("text_recompute")
        if metrics["status"] != 200 or metrics["count"] != 4 or metrics["dimensions"] != [4096] or any(abs(value - 1.0) > 1e-4 for value in metrics["norm_range"]):
            failures.append("text_gate")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        failures.append("text_parse")

    try:
        receipt = receipts["mrl_1024"]
        stored = checks["mrl_1024"]
        if receipt["status"] == 400:
            computed = {
                "status": 400,
                "supported": False,
                "expected_vllm_limitation": True,
                "error": receipt["response"].get("error", {}).get("message", ""),
            }
            if "does not support Matryoshka" not in computed["error"]:
                failures.append("mrl_error")
        else:
            vector = single_vector(receipts, "mrl_1024")
            computed = {"status": receipt["status"], "supported": True, "dimensions": len(vector), "norm": l2(vector)}
            if receipt["status"] != 200 or len(vector) != 1024 or abs(l2(vector) - 1.0) > 1e-4:
                failures.append("mrl_gate")
        if not all(equivalent(stored.get(key), value) for key, value in computed.items()):
            failures.append("mrl_recompute")
    except (KeyError, TypeError, ValueError):
        failures.append("mrl_parse")

    try:
        red_query = single_vector(receipts, "cross_red_query")
        blue_query = single_vector(receipts, "cross_blue_query")
        red_image = single_vector(receipts, "cross_red_image")
        blue_image = single_vector(receipts, "cross_blue_image")
        vectors = [red_query, blue_query, red_image, blue_image]
        matrix = [
            [cosine(red_query, red_image), cosine(red_query, blue_image)],
            [cosine(blue_query, red_image), cosine(blue_query, blue_image)],
        ]
        stored = checks["cross_modal"]
        if not all(finite_vector(vector) for vector in vectors):
            failures.append("cross_raw_vectors")
        if not equivalent(stored.get("similarity_matrix_query_rows_red_blue_image_cols_red_blue"), matrix):
            failures.append("cross_matrix_recompute")
        if stored.get("dimensions") != sorted({len(vector) for vector in vectors}):
            failures.append("cross_dimensions_recompute")
        if not equivalent(stored.get("norm_range"), [min(map(l2, vectors)), max(map(l2, vectors))]):
            failures.append("cross_norm_recompute")
        usages = [receipts[name]["response"].get("usage", {}) for name in ("cross_red_query", "cross_blue_query", "cross_red_image", "cross_blue_image")]
        if stored.get("usage") != usages:
            failures.append("cross_usage_recompute")
        passed = matrix[0][0] - matrix[0][1] > 0.1 and matrix[1][1] - matrix[1][0] > 0.1
        if stored.get("diagonal_retrieval_pass") is not passed or not passed or any(abs(l2(vector) - 1.0) > 1e-4 for vector in vectors):
            failures.append("cross_gate")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        failures.append("cross_parse")

    try:
        ember_query = single_vector(receipts, "ocr_ember_query")
        ocean_query = single_vector(receipts, "ocr_ocean_query")
        ember_image = single_vector(receipts, "ocr_ember_image")
        ocean_image = single_vector(receipts, "ocr_ocean_image")
        vectors = [ember_query, ocean_query, ember_image, ocean_image]
        matrix = [
            [cosine(ember_query, ember_image), cosine(ember_query, ocean_image)],
            [cosine(ocean_query, ember_image), cosine(ocean_query, ocean_image)],
        ]
        stored = checks["controlled_ocr"]
        if not all(finite_vector(vector) for vector in vectors):
            failures.append("ocr_raw_vectors")
        if not equivalent(stored.get("similarity_matrix_query_rows_ember_ocean_image_cols_ember_ocean"), matrix):
            failures.append("ocr_matrix_recompute")
        if stored.get("dimensions") != sorted({len(vector) for vector in vectors}):
            failures.append("ocr_dimensions_recompute")
        if not equivalent(stored.get("norm_range"), [min(map(l2, vectors)), max(map(l2, vectors))]):
            failures.append("ocr_norm_recompute")
        usages = [receipts[name]["response"].get("usage", {}) for name in ("ocr_ember_image", "ocr_ocean_image")]
        if stored.get("image_usage") != usages:
            failures.append("ocr_usage_recompute")
        passed = matrix[0][0] - matrix[0][1] > 0.02 and matrix[1][1] - matrix[1][0] > 0.02
        if stored.get("diagonal_retrieval_pass") is not passed or not passed or any(abs(l2(vector) - 1.0) > 1e-4 for vector in vectors):
            failures.append("ocr_gate")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        failures.append("ocr_parse")

    try:
        names = [f"concurrency_{index}" for index in range(4)]
        vectors = [single_vector(receipts, name) for name in names]
        computed = [
            {"status": receipts[name]["status"], "dimensions": len(vector), "norm": l2(vector)}
            for name, vector in zip(names, vectors)
        ]
        stored = checks["concurrency_4"]
        for index, metrics in enumerate(computed):
            if not all(equivalent(stored[index].get(key), value) for key, value in metrics.items()):
                failures.append(f"concurrency_recompute:{index}")
        overlap = max(receipts[name]["started_unix_ns"] for name in names) < min(receipts[name]["finished_unix_ns"] for name in names)
        if len(stored) != 4 or not overlap or not all(item["status"] == 200 and item["dimensions"] == 4096 and abs(item["norm"] - 1.0) <= 1e-4 for item in computed):
            failures.append("concurrency_gate")
    except (KeyError, IndexError, TypeError, ValueError):
        failures.append("concurrency_parse")

    try:
        query = single_vector(receipts, "long_query")
        positive = single_vector(receipts, "long_positive")
        negative = single_vector(receipts, "long_negative")
        scores = [cosine(query, positive), cosine(query, negative)]
        positive_usage = receipts["long_positive"]["response"].get("usage", {})
        negative_usage = receipts["long_negative"]["response"].get("usage", {})
        stored = checks["long_context_retrieval"]
        if not equivalent(stored.get("positive_similarity"), scores[0]) or not equivalent(stored.get("negative_similarity"), scores[1]):
            failures.append("long_similarity_recompute")
        if stored.get("positive_usage") != positive_usage or stored.get("negative_usage") != negative_usage:
            failures.append("long_usage_recompute")
        passed = scores[0] - scores[1] > 0.05
        if (
            stored.get("needle_retrieval_pass") is not passed
            or not passed
            or positive_usage.get("prompt_tokens", 0) < 25000
            or negative_usage.get("prompt_tokens", 0) < 25000
            or not all(finite_vector(vector) and abs(l2(vector) - 1.0) <= 1e-4 for vector in (query, positive, negative))
        ):
            failures.append("long_gate")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        failures.append("long_parse")

    if data.get("pass") is not True or data.get("failures") != []:
        failures.append("stored_acceptance_status")
    return failures


def validate_nonreg(data):
    failures = []
    try:
        uuid.UUID(data["run_id"])
        if parse_ns(data["timestamp_utc"]) >= parse_ns(data["finished_utc"]):
            failures.append("nonreg_interval")
    except (KeyError, TypeError, ValueError):
        failures.append("nonreg_identity")
    checks = data.get("checks", {})
    for name, marker in EXPECTED_NONREG.items():
        item = checks.get(name, {})
        if not (
            item.get("status") == 200
            and item.get("content", "").strip() == marker
            and item.get("exact_match") is True
            and item.get("finish_reason") == "stop"
        ):
            failures.append(name)
    if set(checks) != set(EXPECTED_NONREG) or data.get("pass") is not True:
        failures.append("nonreg_shape_or_status")
    return failures


def validate_manifest(manifest):
    failures = []
    rows = manifest.get("files", [])
    if not isinstance(rows, list) or {row.get("path") for row in rows if isinstance(row, dict)} != REQUIRED_ARTIFACTS:
        failures.append("required_artifact_set")
        return failures
    for row in rows:
        try:
            path = (ROOT / row["path"]).resolve()
            payload = path.read_bytes()
            if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                failures.append(f"artifact_integrity:{row['path']}")
        except (KeyError, OSError, TypeError):
            failures.append(f"artifact_unreadable:{row.get('path', '<missing>')}")
    return failures


def runtime_receipt_map(runtime):
    return {item["name"]: item for item in runtime.get("receipts", [])}


def command_ok(receipt):
    if receipt.get("returncode") == 0:
        return True
    return receipt.get("name") == "kernel_fatal_scan" and receipt.get("returncode") == 1 and receipt.get("stdout", "").strip() == "-- No entries --"


def validate_runtime(runtime, acceptance, nonreg):
    failures = []
    receipts = runtime_receipt_map(runtime)
    if runtime.get("host") != "192.168.178.51" or runtime.get("container") != "sparkrun_7d7b52c2c9174082_15b33f409193_solo":
        failures.append("runtime_target")
    try:
        gate_ns = max(parse_ns(acceptance["finished_utc"]), parse_ns(nonreg["finished_utc"]))
        if not (
            runtime.get("acceptance_run_id") == acceptance["run_id"]
            and runtime.get("acceptance_result_sha256") == sha256(ROOT / "acceptance-result.json")
            and runtime.get("acceptance_started_utc") == acceptance["timestamp_utc"]
            and runtime.get("acceptance_finished_utc") == acceptance["finished_utc"]
            and runtime.get("non_regression_run_id") == nonreg["run_id"]
            and runtime.get("non_regression_result_sha256") == sha256(ROOT / "non-regression-result.json")
            and runtime.get("non_regression_timestamp_utc") == nonreg["timestamp_utc"]
            and runtime.get("non_regression_finished_utc") == nonreg["finished_utc"]
            and parse_ns(runtime["captured_at_utc"]) > gate_ns
        ):
            failures.append("runtime_run_binding")
    except (KeyError, OSError, TypeError, ValueError):
        failures.append("runtime_run_binding_parse")
        gate_ns = 0

    recipe = REPO / "recipes" / "qwen3-vl-embedding-8b-dgx-spark.yaml"
    recipe_sha = sha256(recipe)
    if recipe_sha != EXPECTED_RECIPE_SHA or runtime.get("recipe_sha256") != recipe_sha:
        failures.append("recipe_sha")
    if set(receipts) != EXPECTED_RECEIPTS or not all(command_ok(item) for item in receipts.values()):
        failures.append("receipt_commands")
    else:
        for name, parts in EXPECTED_COMMAND_PARTS.items():
            command = receipts[name].get("command", "")
            if "ssh -o BatchMode=yes 192.168.178.51" not in command or not all(part in command for part in parts):
                failures.append(f"receipt_binding:{name}")
            if receipts[name].get("started_unix_ns", 0) <= gate_ns or receipts[name].get("finished_unix_ns", 0) <= receipts[name].get("started_unix_ns", 0):
                failures.append(f"receipt_timestamp:{name}")
        if runtime.get("acceptance_started_utc", "") not in receipts["kernel_fatal_scan"].get("command", ""):
            failures.append("kernel_since_binding")
    if receipts.get("host_identity", {}).get("stdout", "").strip() != "gx10":
        failures.append("host_identity")

    http = runtime.get("http", [])
    if len(http) != 2 or not all(item.get("status") == 200 and item.get("started_unix_ns", 0) > gate_ns for item in http):
        failures.append("http_health")
    models = next((item for item in http if item.get("url", "").endswith("/v1/models")), {})
    try:
        row = json.loads(models["body"])["data"][0]
        if row["id"] != MODEL or row["max_model_len"] != 32768:
            failures.append("model_discovery")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        failures.append("model_discovery_parse")

    try:
        container = json.loads(receipts["container_inspect"]["stdout"])[0]
        host = container["HostConfig"]
        labels = container["Config"]["Labels"]
        devices = host["Devices"]
        gpu_requests = host["DeviceRequests"]
        binds = set(host["Binds"])
        required_binds = {
            "/etc/group:/etc/group:ro",
            "/etc/passwd:/etc/passwd:ro",
            "/home/max/.cache/huggingface:/cache/huggingface",
            "/home/max/.cache/sparkrun/runtime-cache/vllm/Qwen__Qwen3-VL-Embedding-8B-9a3f8dcc:/cache/runtime",
        }
        if not (
            container["State"]["Status"] == "running"
            and container["State"]["OOMKilled"] is False
            and container["Image"] == EXPECTED_IMAGE_ID
            and container["Config"]["User"] == "1002:1002"
            and host["Privileged"] is False
            and host["NetworkMode"] == "host"
            and host["IpcMode"] == "host"
            and set(host["SecurityOpt"]) == {"no-new-privileges", "label=disable"}
            and host["CapAdd"] == ["CAP_SYS_PTRACE"]
            and host["ShmSize"] == 34359738368
            and any(item.get("PathOnHost") == "/dev/infiniband" and item.get("PathInContainer") == "/dev/infiniband" and item.get("CgroupPermissions") == "rwm" for item in devices)
            and any(item.get("Count") == -1 and item.get("Capabilities") == [["gpu"]] for item in gpu_requests)
            and required_binds.issubset(binds)
            and labels["sparkrun.cluster_id"] == "sparkrun_7d7b52c2c9174082_15b33f409193"
            and labels["sparkrun.model"] == "Qwen/Qwen3-VL-Embedding-8B"
            and labels["sparkrun.recipe"] == "recipes/qwen3-vl-embedding-8b-dgx-spark.yaml"
            and labels["sparkrun.served_model_name"] == MODEL
        ):
            failures.append("container_security")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        failures.append("container_inspect_parse")

    try:
        image = json.loads(receipts["image_inspect"]["stdout"])[0]
        labels = image["Config"]["Labels"]
        if not (
            image["Id"] == EXPECTED_IMAGE_ID
            and EXPECTED_IMAGE_DIGEST in image["RepoDigests"]
            and image["Architecture"] == "arm64"
            and image["Os"] == "linux"
            and labels["dev.sparkrun.vllm-hash"] == "2902ca17e335457a0fa214638936d154907a2e18"
            and labels["dev.sparkrun.vllm-version"] == "0.28.1rc1.dev441+g2902ca17e.d20260905"
            and labels["dev.sparkrun.repo-commit"] == "94b3c3060133de8f2714e9429f3f8e9154ac83cc"
        ):
            failures.append("image_identity")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        failures.append("image_inspect_parse")

    try:
        versions = json.loads(receipts["runtime_versions"]["stdout"])
        if versions != {
            "cuda": "13.0",
            "nccl": "2.29.7",
            "torch": "2.13.0+cu130",
            "vllm": "0.28.1rc1.dev441+g2902ca17e.d20260905",
        }:
            failures.append("runtime_versions")
    except (KeyError, TypeError, json.JSONDecodeError):
        failures.append("runtime_versions_parse")

    try:
        snapshot = json.loads(receipts["model_snapshot"]["stdout"])
        if not (snapshot["exists"] is True and len(snapshot["shards"]) == 4 and snapshot["missing"] == [] and snapshot["broken_symlinks"] == [] and snapshot["shard_bytes"] == 16289679624):
            failures.append("model_snapshot")
    except (KeyError, TypeError, json.JSONDecodeError):
        failures.append("model_snapshot_parse")

    cache = receipts.get("cache_identity_and_writability", {})
    if cache.get("returncode") != 0 or "uid=1002(max)" not in cache.get("stdout", "") or "max:max 1002:1002" not in cache.get("stdout", ""):
        failures.append("cache_identity")
    process_table = receipts.get("container_processes", {}).get("stdout", "")
    if "vllm serve" not in process_table or "2c4565515e0f265c6511776e7193b22c0968ddc7" not in process_table or "VLLM::EngineCore" not in process_table:
        failures.append("serve_process_binding")
    if "VLLM::EngineCore" not in receipts.get("gpu_processes", {}).get("stdout", ""):
        failures.append("gpu_process")

    try:
        log = json.loads(receipts["serve_log"]["stdout"])
        content = log["content"]
        fatal_patterns = [r"Traceback", r"\bERROR\b", r"Permission denied", r"CUDA error|illegal memory|out of memory", r"NCCL.*(?:error|fail)"]
        if (
            log["path"] != "/tmp/sparkrun_serve.log"
            or log["bytes"] != len(content.encode())
            or log["sha256"] != hashlib.sha256(content.encode()).hexdigest()
            or log["mtime_ns"] < parse_ns(acceptance["timestamp_utc"])
            or content.count("POST /v1/embeddings") < 17
            or any(re.search(pattern, content, re.I) for pattern in fatal_patterns)
        ):
            failures.append("serve_log")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        failures.append("serve_log_parse")
    kernel = receipts.get("kernel_fatal_scan", {})
    if not (kernel.get("returncode") == 1 and kernel.get("stdout", "").strip() == "-- No entries --"):
        failures.append("kernel_fatal")
    return failures


def validate_audit(audit, runtime):
    failures = []
    receipts = runtime_receipt_map(runtime)
    try:
        container = json.loads(receipts["container_inspect"]["stdout"])[0]
        host = container["HostConfig"]
        log = json.loads(receipts["serve_log"]["stdout"])
        security = audit["security"]
        expected_security = {
            "container_user": container["Config"]["User"],
            "privileged": host["Privileged"],
            "security_opt": host["SecurityOpt"],
            "network_mode": host["NetworkMode"],
            "ipc_mode": host["IpcMode"],
            "cap_add": host["CapAdd"],
            "devices": host["Devices"],
            "device_requests": host["DeviceRequests"],
            "binds": host["Binds"],
            "shm_size_bytes": host["ShmSize"],
            "api_authentication": False,
            "remote_media_allowlist": ["inline-media.invalid"],
        }
        if security != expected_security:
            failures.append("audit_security")
        if audit.get("recipe_sha256") != EXPECTED_RECIPE_SHA or audit.get("acceptance_run_id") != runtime.get("acceptance_run_id") or audit.get("non_regression_run_id") != runtime.get("non_regression_run_id"):
            failures.append("audit_binding")
        final_scan = audit["final_log_scan"]
        content = log["content"]
        expected_scan = {
            "bytes": log["bytes"],
            "sha256": log["sha256"],
            "traceback": len(re.findall(r"Traceback", content, re.I)),
            "error_level": len(re.findall(r"\bERROR\b", content, re.I)),
            "permission_denied": len(re.findall(r"Permission denied", content, re.I)),
            "cuda_fatal": len(re.findall(r"CUDA error|illegal memory|out of memory", content, re.I)),
            "nccl_fatal": len(re.findall(r"NCCL.*(?:error|fail)", content, re.I)),
        }
        if final_scan != expected_scan:
            failures.append("audit_log_scan")
        if audit.get("kernel_scan", {}).get("matches") != 0 or audit.get("kernel_scan", {}).get("since_utc") != runtime.get("acceptance_started_utc"):
            failures.append("audit_kernel_scan")
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        failures.append("audit_parse")
    return failures


def load_raw(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    acceptance = json.loads((ROOT / "acceptance-result.json").read_text())
    raw = load_raw(ROOT / "acceptance-raw.json.gz")
    nonreg = json.loads((ROOT / "non-regression-result.json").read_text())
    runtime = json.loads((ROOT / "runtime-receipt.json").read_text())
    audit = json.loads((ROOT / "runtime-audit.json").read_text())
    manifest = json.loads((ROOT / "artifact-manifest.json").read_text())
    baseline = {
        "acceptance": validate_acceptance(acceptance, raw),
        "non_regression": validate_nonreg(nonreg),
        "runtime": validate_runtime(runtime, acceptance, nonreg),
        "runtime_audit": validate_audit(audit, runtime),
        "artifact_manifest": validate_manifest(manifest),
    }

    negative_controls = {}
    mutated = copy.deepcopy(acceptance)
    for section in ("text_batch", "cross_modal", "controlled_ocr"):
        mutated["checks"][section]["norm_range"] = [1.0, 1.0]
    mutated["checks"]["cross_modal"]["similarity_matrix_query_rows_red_blue_image_cols_red_blue"] = [[1.0, 0.0], [0.0, 1.0]]
    mutated["checks"]["controlled_ocr"]["similarity_matrix_query_rows_ember_ocean_image_cols_ember_ocean"] = [[1.0, 0.0], [0.0, 1.0]]
    negative_controls["reject_metric_substitution"] = bool(validate_acceptance(mutated, raw))
    mutated_raw = copy.deepcopy(raw)
    receipt_map(mutated_raw)["text_batch"]["response"]["data"][0]["embedding"][0] += 0.25
    negative_controls["reject_raw_vector_substitution"] = bool(validate_acceptance(acceptance, mutated_raw))
    mutated_raw = copy.deepcopy(raw)
    receipt_map(mutated_raw)["cross_red_image"]["status"] = 500
    negative_controls["reject_failed_cross_modal_request"] = bool(validate_acceptance(acceptance, mutated_raw))
    mutated_raw = copy.deepcopy(raw)
    receipt_map(mutated_raw)["long_positive"]["response"]["model"] = "substituted-model"
    negative_controls["reject_response_model_substitution"] = bool(validate_acceptance(acceptance, mutated_raw))
    mutated_raw = copy.deepcopy(raw)
    receipt_map(mutated_raw)["long_query"]["request"]["messages"][0]["content"][0]["text"] = "substituted request"
    negative_controls["reject_raw_request_substitution"] = bool(validate_acceptance(acceptance, mutated_raw))
    mutated_raw = copy.deepcopy(raw)
    concurrent = [receipt_map(mutated_raw)[f"concurrency_{index}"] for index in range(4)]
    for index, receipt in enumerate(concurrent):
        receipt["started_unix_ns"] = index * 10 + 1
        receipt["finished_unix_ns"] = index * 10 + 2
    negative_controls["reject_false_concurrency"] = bool(validate_acceptance(acceptance, mutated_raw))
    mutated = copy.deepcopy(nonreg)
    mutated["checks"]["glm_direct"]["content"] = "EXTRA GLM-ALIVE-531 EXTRA"
    negative_controls["reject_non_exact_completion"] = bool(validate_nonreg(mutated))
    mutated = copy.deepcopy(runtime)
    container = json.loads(runtime_receipt_map(mutated)["container_inspect"]["stdout"])
    container[0]["HostConfig"]["Privileged"] = True
    runtime_receipt_map(mutated)["container_inspect"]["stdout"] = json.dumps(container)
    negative_controls["reject_privileged_container"] = bool(validate_runtime(mutated, acceptance, nonreg))
    mutated = copy.deepcopy(runtime)
    container = json.loads(runtime_receipt_map(mutated)["container_inspect"]["stdout"])
    container[0]["HostConfig"]["CapAdd"] = []
    runtime_receipt_map(mutated)["container_inspect"]["stdout"] = json.dumps(container)
    negative_controls["reject_security_surface_substitution"] = bool(validate_runtime(mutated, acceptance, nonreg))
    mutated = copy.deepcopy(runtime)
    mutated["acceptance_run_id"] = str(uuid.uuid4())
    negative_controls["reject_run_relabeling"] = bool(validate_runtime(mutated, acceptance, nonreg))
    mutated = copy.deepcopy(runtime)
    for receipt in mutated["receipts"]:
        receipt["started_unix_ns"] = 1
        receipt["finished_unix_ns"] = 2
    negative_controls["reject_predated_runtime"] = bool(validate_runtime(mutated, acceptance, nonreg))
    mutated = copy.deepcopy(manifest)
    mutated["files"] = [row for row in mutated["files"] if row["path"] != "acceptance-raw.json.gz"]
    negative_controls["reject_required_raw_deletion"] = bool(validate_manifest(mutated))
    mutated = copy.deepcopy(manifest)
    next(row for row in mutated["files"] if row["path"] == "runtime-receipt.json")["path"] = "renamed-runtime.json"
    negative_controls["reject_artifact_relabeling"] = bool(validate_manifest(mutated))

    passed = all(not failures for failures in baseline.values()) and all(negative_controls.values())
    result = {"baseline_failures": baseline, "negative_controls": negative_controls, "pass": passed}
    (ROOT / "verification-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
