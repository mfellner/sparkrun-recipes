#!/usr/bin/env python3
"""Deterministic direct-API acceptance for Qwen3-VL-Embedding-8B."""

import argparse
import base64
import concurrent.futures
import gzip
import hashlib
import json
import math
import pathlib
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any

from PIL import Image, ImageDraw, ImageFont

MODEL = "qwen3-vl-embedding-8b"
INSTRUCTION = "Retrieve images or text relevant to the user's query."
TEXT_BATCH = [
    "A red circle marked EMBER-742.",
    "A blue square marked OCEAN-319.",
    "A golden retriever running on a beach.",
    "A database migration using PostgreSQL.",
]
MRL_INPUT = "A compact vector dimension check."
RED_QUERY = "Retrieve the image with a red circle on the left and code EMBER-742."
BLUE_QUERY = "Retrieve the image with a blue square on the right and code OCEAN-319."
EMBER_OCR_QUERY = "Retrieve the image displaying token EMBER-742."
OCEAN_OCR_QUERY = "Retrieve the image displaying token OCEAN-319."
CONCURRENT_INPUTS = [f"Concurrent embedding request {index}: cobalt archive." for index in range(4)]
BOILERPLATE_UNIT = "Telemetry nominal. Cooling stable. Power rails stable. "
BOILERPLATE_REPEATS = 1200
LONG_QUERY = "Retrieve the report containing maintenance token ORBIT-991 and a replaced coolant valve."
LONG_POSITIVE_NEEDLE = " Maintenance token ORBIT-991 confirms the coolant valve was replaced. "
LONG_NEGATIVE_NEEDLE = " Maintenance token LUNAR-224 confirms the intake filter was inspected. "


def post_json(url, payload, timeout=300, allow_http_error=False):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
            return response.status, data, time.monotonic() - started
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        if allow_http_error:
            return exc.code, json.loads(detail), time.monotonic() - started
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def l2(vector):
    return math.sqrt(sum(value * value for value in vector))


def cosine(left, right):
    return sum(a * b for a, b in zip(left, right)) / (l2(left) * l2(right))


def unit_norm(value, tolerance=1e-4):
    return math.isfinite(value) and abs(value - 1.0) <= tolerance


def font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_fixture(path, shape, color, code, side):
    background: Any = (247, 243, 232)
    image = Image.new("RGB", (640, 420), background)
    draw = ImageDraw.Draw(image)
    if shape == "circle":
        draw.ellipse((55, 55, 315, 315), fill=color, outline="#202020", width=8)
    else:
        draw.rectangle((325, 55, 585, 315), fill=color, outline="#202020", width=8)
    draw.text((34, 336), f"{shape.upper()}  {code}", fill="#111111", font=font(42))
    draw.text((430 if side == "right" else 34, 16), side.upper(), fill="#333333", font=font(28))
    image.save(path, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def make_ocr_fixture(path, code):
    background: Any = (238, 238, 238)
    image = Image.new("RGB", (640, 420), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((45, 55, 595, 350), radius=24, fill="#ffffff", outline="#303030", width=8)
    draw.text((115, 105), "MAINTENANCE TOKEN", fill="#111111", font=font(36))
    draw.text((145, 220), code, fill="#111111", font=font(56))
    image.save(path, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def chat_embedding(requester, name, text="", image_uri=None, instruction=INSTRUCTION):
    content = []
    if image_uri:
        content.append({"type": "image_url", "image_url": {"url": image_uri}})
    content.append({"type": "text", "text": text})
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ],
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
    }
    status, response, elapsed = requester(name, payload)
    return response["data"][0]["embedding"], response.get("usage", {}), status, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.178.51:8000")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = pathlib.Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    red_path = output_path.parent / "red-circle-EMBER-742.png"
    blue_path = output_path.parent / "blue-square-OCEAN-319.png"
    ember_ocr_path = output_path.parent / "ocr-card-EMBER-742.png"
    ocean_ocr_path = output_path.parent / "ocr-card-OCEAN-319.png"
    red_uri = make_fixture(red_path, "circle", "#d52b1e", "EMBER-742", "left")
    blue_uri = make_fixture(blue_path, "square", "#1769d1", "OCEAN-319", "right")
    ember_ocr_uri = make_ocr_fixture(ember_ocr_path, "EMBER-742")
    ocean_ocr_uri = make_ocr_fixture(ocean_ocr_path, "OCEAN-319")
    endpoint = args.base_url.rstrip("/") + "/v1/embeddings"
    run_id = str(uuid.uuid4())
    started_utc = datetime.now(timezone.utc).isoformat()
    receipts = []
    receipts_lock = threading.Lock()

    def record_post(name, payload, timeout=300, allow_http_error=False):
        started_unix_ns = time.time_ns()
        status, response, elapsed = post_json(
            endpoint,
            payload,
            timeout=timeout,
            allow_http_error=allow_http_error,
        )
        finished_unix_ns = time.time_ns()
        with receipts_lock:
            receipts.append(
                {
                    "name": name,
                    "started_unix_ns": started_unix_ns,
                    "finished_unix_ns": finished_unix_ns,
                    "status": status,
                    "request": payload,
                    "response": response,
                }
            )
        return status, response, elapsed

    result = {
        "timestamp_utc": started_utc,
        "run_id": run_id,
        "base_url": args.base_url,
        "model": MODEL,
        "fixtures": {
            red_path.name: hashlib.sha256(red_path.read_bytes()).hexdigest(),
            blue_path.name: hashlib.sha256(blue_path.read_bytes()).hexdigest(),
            ember_ocr_path.name: hashlib.sha256(ember_ocr_path.read_bytes()).hexdigest(),
            ocean_ocr_path.name: hashlib.sha256(ocean_ocr_path.read_bytes()).hexdigest(),
        },
        "request_controls": {
            "instruction": INSTRUCTION,
            "text_batch": TEXT_BATCH,
            "mrl_input": MRL_INPUT,
            "visual_queries": [RED_QUERY, BLUE_QUERY],
            "controlled_ocr_queries": [EMBER_OCR_QUERY, OCEAN_OCR_QUERY],
            "concurrency_inputs": CONCURRENT_INPUTS,
            "boilerplate_unit": BOILERPLATE_UNIT,
            "boilerplate_repeats_per_side": BOILERPLATE_REPEATS,
            "long_query": LONG_QUERY,
            "long_positive_needle": LONG_POSITIVE_NEEDLE,
            "long_negative_needle": LONG_NEGATIVE_NEEDLE,
        },
        "checks": {},
    }

    # Standard OpenAI-compatible text batching.
    status, response, elapsed = record_post(
        "text_batch",
        {"model": MODEL, "input": TEXT_BATCH, "encoding_format": "float"},
    )
    vectors = [item["embedding"] for item in sorted(response["data"], key=lambda item: item["index"])]
    result["checks"]["text_batch"] = {
        "status": status,
        "count": len(vectors),
        "dimensions": sorted({len(vector) for vector in vectors}),
        "norm_range": [min(map(l2, vectors)), max(map(l2, vectors))],
        "elapsed_s": elapsed,
        "usage": response.get("usage", {}),
    }

    # Matryoshka output dimension support.
    status, response, elapsed = record_post(
        "mrl_1024",
        {"model": MODEL, "input": MRL_INPUT, "dimensions": 1024},
        allow_http_error=True,
    )
    if status == 200:
        mrl_vector = response["data"][0]["embedding"]
        result["checks"]["mrl_1024"] = {
            "status": status,
            "supported": True,
            "dimensions": len(mrl_vector),
            "norm": l2(mrl_vector),
            "elapsed_s": elapsed,
            "usage": response.get("usage", {}),
        }
    else:
        message = response.get("error", {}).get("message", "")
        result["checks"]["mrl_1024"] = {
            "status": status,
            "supported": False,
            "expected_vllm_limitation": status == 400 and "does not support Matryoshka" in message,
            "error": message,
            "elapsed_s": elapsed,
        }

    # Cross-modal retrieval with visual shape/color and unique OCR codes.
    red_query, red_q_usage, _, red_q_elapsed = chat_embedding(record_post, "cross_red_query", RED_QUERY)
    blue_query, blue_q_usage, _, blue_q_elapsed = chat_embedding(record_post, "cross_blue_query", BLUE_QUERY)
    red_image, red_i_usage, _, red_i_elapsed = chat_embedding(record_post, "cross_red_image", image_uri=red_uri)
    blue_image, blue_i_usage, _, blue_i_elapsed = chat_embedding(record_post, "cross_blue_image", image_uri=blue_uri)
    matrix = [
        [cosine(red_query, red_image), cosine(red_query, blue_image)],
        [cosine(blue_query, red_image), cosine(blue_query, blue_image)],
    ]
    cross_modal_pass = (
        matrix[0][0] - matrix[0][1] > 0.1
        and matrix[1][1] - matrix[1][0] > 0.1
        and all(math.isfinite(value) for row in matrix for value in row)
    )
    result["checks"]["cross_modal"] = {
        "similarity_matrix_query_rows_red_blue_image_cols_red_blue": matrix,
        "diagonal_retrieval_pass": cross_modal_pass,
        "dimensions": sorted({len(red_query), len(blue_query), len(red_image), len(blue_image)}),
        "norm_range": [
            min(map(l2, [red_query, blue_query, red_image, blue_image])),
            max(map(l2, [red_query, blue_query, red_image, blue_image])),
        ],
        "usage": [red_q_usage, blue_q_usage, red_i_usage, blue_i_usage],
        "elapsed_s": [red_q_elapsed, blue_q_elapsed, red_i_elapsed, blue_i_elapsed],
    }

    # Controlled OCR retrieval: the two cards are pixel-identical except for
    # the rendered token, so visual shape/color/position cannot explain rank.
    ember_query, _, _, _ = chat_embedding(record_post, "ocr_ember_query", EMBER_OCR_QUERY)
    ocean_query, _, _, _ = chat_embedding(record_post, "ocr_ocean_query", OCEAN_OCR_QUERY)
    ember_card, ember_usage, _, _ = chat_embedding(record_post, "ocr_ember_image", image_uri=ember_ocr_uri)
    ocean_card, ocean_usage, _, _ = chat_embedding(record_post, "ocr_ocean_image", image_uri=ocean_ocr_uri)
    ocr_matrix = [
        [cosine(ember_query, ember_card), cosine(ember_query, ocean_card)],
        [cosine(ocean_query, ember_card), cosine(ocean_query, ocean_card)],
    ]
    ocr_pass = (
        ocr_matrix[0][0] - ocr_matrix[0][1] > 0.02
        and ocr_matrix[1][1] - ocr_matrix[1][0] > 0.02
        and all(math.isfinite(value) for row in ocr_matrix for value in row)
    )
    ocr_vectors = [ember_query, ocean_query, ember_card, ocean_card]
    result["checks"]["controlled_ocr"] = {
        "similarity_matrix_query_rows_ember_ocean_image_cols_ember_ocean": ocr_matrix,
        "diagonal_retrieval_pass": ocr_pass,
        "dimensions": sorted({len(vector) for vector in ocr_vectors}),
        "norm_range": [min(map(l2, ocr_vectors)), max(map(l2, ocr_vectors))],
        "image_usage": [ember_usage, ocean_usage],
    }

    # Four simultaneous requests exercise the live scheduler.
    def concurrent_one(index_and_text):
        index, text = index_and_text
        status, body, elapsed = record_post(f"concurrency_{index}", {"model": MODEL, "input": text})
        vector = body["data"][0]["embedding"]
        return {"status": status, "dimensions": len(vector), "norm": l2(vector), "elapsed_s": elapsed}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_results = list(pool.map(concurrent_one, enumerate(CONCURRENT_INPUTS)))
    result["checks"]["concurrency_4"] = concurrent_results

    # A multi-thousand-token retrieval check, with one distinguishing needle.
    boilerplate = BOILERPLATE_UNIT * BOILERPLATE_REPEATS
    long_query, _, _, _ = chat_embedding(record_post, "long_query", LONG_QUERY)
    positive, positive_usage, _, positive_elapsed = chat_embedding(
        record_post,
        "long_positive",
        boilerplate + LONG_POSITIVE_NEEDLE + boilerplate,
    )
    negative, negative_usage, _, negative_elapsed = chat_embedding(
        record_post,
        "long_negative",
        boilerplate + LONG_NEGATIVE_NEEDLE + boilerplate,
    )
    positive_score = cosine(long_query, positive)
    negative_score = cosine(long_query, negative)
    result["checks"]["long_context_retrieval"] = {
        "positive_similarity": positive_score,
        "negative_similarity": negative_score,
        "needle_retrieval_pass": positive_score > negative_score,
        "positive_usage": positive_usage,
        "negative_usage": negative_usage,
        "elapsed_s": [positive_elapsed, negative_elapsed],
    }

    failures = []
    if result["checks"]["text_batch"]["count"] != 4:
        failures.append("text_batch_count")
    if result["checks"]["text_batch"]["dimensions"] != [4096]:
        failures.append("text_dimensions")
    if not all(unit_norm(l2(vector)) for vector in vectors):
        failures.append("text_norms")
    mrl = result["checks"]["mrl_1024"]
    if not (
        (mrl["supported"] and mrl["dimensions"] == 1024 and unit_norm(mrl["norm"]))
        or mrl.get("expected_vllm_limitation")
    ):
        failures.append("mrl_behavior")
    if not cross_modal_pass:
        failures.append("cross_modal_retrieval")
    if not all(unit_norm(l2(vector)) for vector in [red_query, blue_query, red_image, blue_image]):
        failures.append("cross_modal_norms")
    if not ocr_pass or result["checks"]["controlled_ocr"]["dimensions"] != [4096]:
        failures.append("controlled_ocr_retrieval")
    if not all(unit_norm(l2(vector)) for vector in ocr_vectors):
        failures.append("controlled_ocr_norms")
    if len(concurrent_results) != 4 or any(
        item["status"] != 200 or item["dimensions"] != 4096 or not unit_norm(item["norm"])
        for item in concurrent_results
    ):
        failures.append("concurrency")
    long_check = result["checks"]["long_context_retrieval"]
    if not (
        long_check["needle_retrieval_pass"]
        and long_check["positive_similarity"] - long_check["negative_similarity"] > 0.05
        and long_check["positive_usage"].get("prompt_tokens", 0) >= 25000
        and long_check["negative_usage"].get("prompt_tokens", 0) >= 25000
    ):
        failures.append("long_context_retrieval")
    result["pass"] = not failures
    result["failures"] = failures
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    raw_path = output_path.with_name("acceptance-raw.json.gz")
    raw_payload = {
        "schema": 1,
        "run_id": run_id,
        "started_utc": started_utc,
        "finished_utc": result["finished_utc"],
        "endpoint": endpoint,
        "model": MODEL,
        "receipts": sorted(receipts, key=lambda item: item["name"]),
    }
    with gzip.open(raw_path, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(raw_payload, handle, separators=(",", ":"))
    result["raw_artifact"] = raw_path.name
    result["raw_bytes"] = raw_path.stat().st_size
    result["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
