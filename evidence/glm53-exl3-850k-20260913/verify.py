#!/usr/bin/env python3
"""Recompute the GLM-5.3 EXL3 850K acceptance verdict."""
from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import math
import re
import shlex
import struct
import sys
import zlib
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from command_contract import (  # noqa: E402
    reviewed_recipe_command,
    runtime_command as derived_runtime_command,
    wrapper_command as derived_wrapper_command,
)


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MODEL = "GLM-5.3-Flash-EXL3"
EXPECTED_SYSTEM_FINGERPRINT = "vllm-0.1.dev20051+g487ecf187-tp2-c9097fa8"
CLUSTER = "sparkrun_f906ee990596486e_20260913c411"
EXPECTED_ACCEPTANCE_RUN_ID = "f906c41120260913"
EXPECTED_LAUNCH_ARGV = [
    "sparkrun",
    "run",
    "recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml",
    "--cluster",
    "vacation-pair2",
    "--trust",
    "--no-follow",
    "--container-name",
    CLUSTER,
]
EXPECTED_ACCEPTANCE_ARGV = [
    "evidence/glm53-exl3-850k-20260913/acceptance.py",
    "--direct-url", "http://127.0.0.1:8000",
    "--proxy-url", "http://127.0.0.1:4000",
    "--recipe", "recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml",
    "--cluster-id", CLUSTER,
    "--run-id", EXPECTED_ACCEPTANCE_RUN_ID,
    "--out", "evidence/glm53-exl3-850k-20260913/acceptance.json",
    "--telemetry-signal", "/tmp/glm53-exl3-850k-telemetry.signal",
    "--long-repeat", "55000",
]
IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:eecb36e14dc34c92d46827fde7b09f7e0bf27e27c426ece126376c02dea6cd2f"
RECIPE = REPO / "recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml"
MOD = REPO / "mods/glm-5.3-flash-exl3-upstream-850k"
EXPECTED_MOD_MANIFEST_SHA256 = "4492c3862e7f38fe38505ab4e9e9d5acd7d53ff680be759542c1200aac1f4811"
EXPECTED_HOST_ORDER = ("192.168.178.47", "192.168.178.46")
EXPECTED_HOSTS = set(EXPECTED_HOST_ORDER)
EXPECTED_CHECKS = {
    "direct_models",
    "direct_exact",
    "direct_concurrency_c4",
    "direct_long_context",
    "direct_telemetry_load",
    "direct_reasoning_open_stop",
    "direct_thinking_disabled_stop",
    "direct_tool_call",
    "direct_vision",
    "direct_ocr",
    "direct_remote_media_rejected",
    "direct_video_rejected",
    "proxy_models",
    "proxy_exact",
    "proxy_vision",
    "proxy_ocr",
    "proxy_remote_media_rejected",
    "proxy_video_rejected",
}
EXPECTED_MANIFEST = {
    "acceptance.json",
    "acceptance.py",
    "capture_runtime.py",
    "capture_load_telemetry.py",
    "launch-epoch.txt",
    "launch-receipt.json",
    "launch.log",
    "load-telemetry.log",
    "proxy-models.json",
    "proxy-start.log",
    "README.md",
    "runtime.json",
    "static-validation.log",
    "test_negative_controls.py",
    "verify.py",
    "vision-quadrants.png",
    "vision-ocr.png",
    "video-tiny.gif",
    "command_contract.py",
}
FIXTURE_SHA256 = "8b2fc0401dba5a125ac114d6e98c610fe10128a0d0a405ef8744931eea6c0e86"
OCR_FIXTURE_SHA256 = "75bb474d713ec5d4a951a6dd51b5b6ce59adbb043f9252536bc0e3a567e3cdc5"
IMMUTABLE_DEFAULTS = {
    "port": 8000,
    "served_model_name": MODEL,
    "tensor_parallel": 2,
    "pipeline_parallel": 1,
    "gpu_memory_utilization": 0.85,
    "max_model_len": 850000,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 7168,
}
NCCL_PATH = "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2"
NCCL_SHA256 = "fc7ea66334edbc934aa25959b9907dbb2b91a1d2485beff18839afc45cbc08d0"
E3_EXECUTION_MARKER = "[glm53-e3-executed]"
REASONING_STOP_REASON = 154827
IMAGE_CONFIG_DIGEST = "sha256:9581c4c7425786be27a7904a78c01bb27ae138e33840e605ecc706e76c761900"
EXPECTED_BINDS = {
    "/home/max/.cache/huggingface:/cache/huggingface",
    "/home/max/.cache/sparkrun/runtime-cache/vllm/Mia-AiLab__GLM-5.3-Flash-EXL3-TR3-4bpw-15f54ccc:/cache/runtime",
}
EXPECTED_MOUNTS = {
    ("bind", "/home/max/.cache/huggingface", "/cache/huggingface", True, "", "rprivate"),
    (
        "bind",
        "/home/max/.cache/sparkrun/runtime-cache/vllm/Mia-AiLab__GLM-5.3-Flash-EXL3-TR3-4bpw-15f54ccc",
        "/cache/runtime",
        True,
        "",
        "rprivate",
    ),
}
REMOTE_MEDIA_ERROR = {
    "error": {
        "message": "The URL must be from one of the allowed domains: ['media.invalid']. Input URL domain: 127.0.0.1",
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    }
}
PROXY_REMOTE_MEDIA_ERROR = {
    "error": {
        "code": "400",
        "message": (
            "litellm.BadRequestError: OpenAIException - The URL must be from one of the allowed domains: "
            "['media.invalid']. Input URL domain: 127.0.0.1. Received Model Group=GLM-5.3-Flash-EXL3\n"
            "Available Model Group Fallbacks=None"
        ),
        "param": None,
        "type": None,
    }
}
VIDEO_LIMIT_MESSAGE = (
    "At most 0 video(s) may be provided in one prompt. "
    "Set `--limit-mm-per-prompt` to increase this limit. (parameter=video)"
)
VIDEO_LIMIT_ERROR = {
    "error": {
        "message": VIDEO_LIMIT_MESSAGE,
        "type": "BadRequestError",
        "param": "video",
        "code": 400,
    }
}
PROXY_VIDEO_LIMIT_ERROR = {
    "error": {
        "code": "400",
        "message": (
            "litellm.BadRequestError: OpenAIException - " + VIDEO_LIMIT_MESSAGE
            + ". Received Model Group=GLM-5.3-Flash-EXL3\n"
            "Available Model Group Fallbacks=None"
        ),
        "param": None,
        "type": None,
    }
}
VIDEO_FIXTURE_SHA256 = "b89ebcdbedac896cc0ee9645f92000a1780d2095cfe76aa7a0b2cd38ef93b5f0"
VIDEO_FIXTURE_ZLIB_B64 = (
    "eNpz93SzsEx8wPCAoZGB4T8DAij+5/ZzDQl2dgxwNdIzYGYECf1kYUgB0jogeZAWBo7/DIwcMh4bFhxs5hDWijmxYeHhdgFlrzkeGxcd7ZYwzrpzYuPi4/0Kzl0ynpuWnJysEbwq5uSmpaenGySfmuO5ednZ2RbFr+6c3Lz8/HyHZi5Zry0rLi72mKwVe2rLysvLAxZ7zfXauurq6ojNWXdPbV19fX3C4S5Z721rbm7OuLwq9vS2tbe3Fzw+Ndd7+7q7uys+v7p7evv6+/sbmLnlfHZseHi4Q1g77syOjY+PT1D2nuezc9PT0zOMs++d2bn5+fkFzt1yvru2vLy8Inh13NldW19f35B8ep7v7m1vb+8ofn3v7O7t7+8faOaW99uz4+PjE5O148/t2fn5+YXF3vP99u76+vrG5uz75/bu/v7+weHuAnn/fXt+fn5xeXX8+X17f3//8Pj0fP/9+/7+/vH59f3z+/f//8/ApvGggUUs40Ejh9qKB008Zi8eNAu4aTxsEQnLeNgqkbbiYZtM2YuH7QptGo86VKZlPOrUWLbiUZfOthePug2OaTzuMbmW8bjX4tmKx302WaICoEhgTGFgREQCSryNRgjdI8QaAKLzwUc="
)
# Filled only after a fresh launch/capture. The exact reviewed git tree and this verifier are the external trust root.
# verify.py and SHA256SUMS are intentionally excluded to avoid verifier/self-manifest
# hash circularity.
FINAL_ARTIFACT_SHA256 = {
    "acceptance.py": "13db35afcc47b7aff8ed0166a8eb32b08867a2cc5af4d13384b6d15c1e82cc1c",
    "capture_runtime.py": "029e00e69e777929464ff32c42e7b3076f2396748d58842a3fd69c8f0162fac0",
    "capture_load_telemetry.py": "7aa75e4440c2e550b6a955f1fe217e27d5b7b07a03258b95c3d5b90c9359295a",
    "command_contract.py": "c00e8c9b4cf6a752eda1380b74bcabb6aaf1c02c17d9b7f6d719cecf1cab3eaa",
    "acceptance.json": "683caf37dec9222bbf4aa7e279bb6c4dbdc1cfdee8a5e7a39aa1a10a21dc10cb",
    "runtime.json": "7d1981839737b5490dfc32085c9c7244dbd6aab5a1d4f8956215b33c05e987af",
    "load-telemetry.log": "27dfdc8894055255c94cc7f034c2b3681497d5d1173a38e6c886e05f339c8841",
    "proxy-models.json": "c29566ce4608d94b8be92030d313056f6476a9b794198bcdbc4013f2cba9cac5",
    "proxy-start.log": "8e0c2f59545f0a1cfd9b31852a52ce53002930f997469a98e1b5143cc703c0c8",
    "launch-epoch.txt": "e069dee425cfe2497961fa9b294b971f6659e1fbb5930507230747129950116e",
    "launch-receipt.json": "8339da7f0518bcccb44443b634ec1ca63fc1cc091162e359717ae7e3d188e95b",
    "launch.log": "6ed131ad3aed8007b255d990a6de54e084af7dc4fba8a0d60d3dcea0cad81637",
    "vision-quadrants.png": "8b2fc0401dba5a125ac114d6e98c610fe10128a0d0a405ef8744931eea6c0e86",
    "vision-ocr.png": "75bb474d713ec5d4a951a6dd51b5b6ce59adbb043f9252536bc0e3a567e3cdc5",
    "video-tiny.gif": "b89ebcdbedac896cc0ee9645f92000a1780d2095cfe76aa7a0b2cd38ef93b5f0",
    "README.md": "4868e678c9079be81b1f20c0b69d7939171badff17437a1dc7ef46e426aa8142",
    "static-validation.log": "ff0f501fb26a7317fa043b711ff634fcd68c444b08e548b2c1dc3cf24274a725",
}


def artifact_binding_failures(root: Path, bindings: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for rel, expected in bindings.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            failures.append(f"final artifact binding pending: {rel}")
            continue
        path = root / rel
        if not path.is_file() or sha(path) != expected:
            failures.append(f"final artifact exact hash: {rel}")
    return failures


def expected_capture_argv(launch_epoch: int, acceptance_run_id: str) -> list[str]:
    return [
        "evidence/glm53-exl3-850k-20260913/capture_runtime.py",
        "--cluster-id", CLUSTER,
        "--launch-epoch", str(launch_epoch),
        "--launch-receipt", "evidence/glm53-exl3-850k-20260913/launch-receipt.json",
        "--acceptance", "evidence/glm53-exl3-850k-20260913/acceptance.json",
        "--acceptance-run-id", acceptance_run_id,
        "--expected-mod-manifest-sha256", EXPECTED_MOD_MANIFEST_SHA256,
        "--out", "evidence/glm53-exl3-850k-20260913/runtime.json",
    ]


def top_level_metadata_failures(
    acceptance: object,
    runtime: object,
    launch_epoch: object,
    expected_run_id: str,
) -> list[str]:
    failures: list[str] = []
    acceptance_row = acceptance if isinstance(acceptance, dict) else {}
    runtime_row = runtime if isinstance(runtime, dict) else {}
    if not (
        type(acceptance_row.get("schema")) is int
        and acceptance_row.get("schema") == 1
        and acceptance_row.get("process_role") == "exact_final"
        and acceptance_row.get("passed") is True
    ):
        failures.append("acceptance schema/process/verdict")
    if expected_run_id == "PENDING_RECAPTURE":
        failures.append("acceptance run ID pending recapture")
    elif (
        re.fullmatch(r"[0-9a-f]{16}", expected_run_id) is None
        or acceptance_row.get("run_id") != expected_run_id
    ):
        failures.append("acceptance exact run ID")
    if not (
        type(runtime_row.get("schema")) is int
        and runtime_row.get("schema") == 1
        and runtime_row.get("process_role") == "exact_final"
        and runtime_row.get("producer") == "capture_runtime.py"
    ):
        failures.append("runtime schema/process/producer")
    if runtime_row.get("acceptance_run_id") != acceptance_row.get("run_id"):
        failures.append("runtime/acceptance run ID join")
    expected_epoch = launch_epoch if isinstance(launch_epoch, int) else -1
    if runtime_row.get("capture_argv") != expected_capture_argv(
        expected_epoch, expected_run_id
    ):
        failures.append("runtime capture argv")
    return failures


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def command_binding_failures(
    contract: object, status_raw_command: object, launch_receipt: object
) -> list[str]:
    reason = "reviewed recipe/SparkRun/launch command binding"
    if not isinstance(contract, dict) or not isinstance(launch_receipt, dict):
        return [reason]
    reviewed = contract.get("raw_command")
    if (
        not isinstance(reviewed, str)
        or status_raw_command != reviewed
        or launch_receipt.get("recipe_command") != reviewed
    ):
        return [reason]
    return []


def launch_epoch_failures(
    launch_epoch: object,
    receipt: object,
    docker_facts: object,
    recipe_sha256: str,
    contract: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    expected_receipt_keys = {
        "schema", "cluster_id", "recipe_sha256", "recipe_command",
        "mod_manifest_sha256", "launch_epoch", "recorded_at", "argv"
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_receipt_keys:
        return ["immutable launch receipt schema"]
    try:
        recorded_epoch = int(timestamp(receipt.get("recorded_at")))
    except (TypeError, ValueError):
        recorded_epoch = -1
    if (
        receipt.get("schema") != 1
        or receipt.get("cluster_id") != CLUSTER
        or receipt.get("recipe_sha256") != recipe_sha256
        or receipt.get("recipe_command") != contract.get("raw_command")
        or receipt.get("launch_epoch") != launch_epoch
        or receipt.get("argv") != EXPECTED_LAUNCH_ARGV
        or recorded_epoch != launch_epoch
    ):
        failures.append("immutable launch receipt binding")
    if receipt.get("mod_manifest_sha256") != EXPECTED_MOD_MANIFEST_SHA256:
        failures.append("launch mod manifest binding")
    if not isinstance(launch_epoch, int) or not isinstance(docker_facts, dict):
        failures.append("launch epoch precedes Docker Created")
        return failures
    created_values: list[float] = []
    docker_valid = set(docker_facts) == EXPECTED_HOSTS
    for host in EXPECTED_HOSTS:
        facts = docker_facts.get(host, {})
        try:
            created = timestamp(facts["Created"])
            started = timestamp(facts["StartedAt"])
            created_values.append(created)
            docker_valid = docker_valid and launch_epoch <= created <= started
            docker_valid = docker_valid and created - launch_epoch <= 300
        except (KeyError, TypeError, ValueError):
            docker_valid = False
    if not docker_valid:
        failures.append("launch epoch precedes Docker Created")
    if len(created_values) == len(EXPECTED_HOSTS) and max(created_values) - min(created_values) > 5:
        failures.append("cross-host Docker Created window")
    return failures


def message(row: dict) -> dict:
    try:
        return row["response"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}


def reasoning_stop_failures(choice: object) -> list[str]:
    failures: list[str] = []
    row = choice if isinstance(choice, dict) else {}
    response_message = row.get("message")
    response_message = response_message if isinstance(response_message, dict) else {}
    if "144" not in (response_message.get("reasoning") or ""):
        failures.append("reasoning-open client stop appeared in reasoning")
    if response_message.get("content") != "GLM53_REASONING_STOP_OK":
        failures.append("reasoning-open exact final answer")
    if row.get("finish_reason") != "stop":
        failures.append("reasoning-open finish reason")
    if row.get("stop_reason") != REASONING_STOP_REASON:
        failures.append("reasoning-open exact stop reason")
    return failures


def thinking_disabled_stop_failures(choice: object) -> list[str]:
    failures: list[str] = []
    row = choice if isinstance(choice, dict) else {}
    response_message = row.get("message")
    response_message = response_message if isinstance(response_message, dict) else {}
    content = response_message.get("content")
    if content != "BEFORE ":
        failures.append("thinking-disabled stop truncated output")
    if row.get("finish_reason") != "stop":
        failures.append("thinking-disabled finish reason")
    if row.get("stop_reason") != "Question:":
        failures.append("thinking-disabled client stop reason")
    return failures


def require(condition: bool, reason: str, failures: list[str]) -> None:
    if not condition:
        failures.append(reason)


def request_sha(body: dict) -> str:
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


def exact_request(marker: str) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Reply with exactly {marker} and nothing else."}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def canonical_fixture() -> bytes:
    rows = []
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    for y in range(256):
        row = bytearray([0])
        for x in range(256):
            index = (2 if y >= 128 else 0) + (1 if x >= 128 else 0)
            row.extend(colors[index])
        rows.append(bytes(row))
    ihdr = struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + png_chunk(b"IEND", b"")


OCR_TEXT = "GLM53 OCR 8429"
OCR_GLYPHS = {
    " ": ("00000",) * 7,
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
}


def canonical_ocr_fixture() -> bytes:
    width, height, scale, left, top = 544, 96, 6, 14, 27
    pixels = bytearray([255]) * (width * height * 3)
    x0 = left
    for character in OCR_TEXT:
        glyph = OCR_GLYPHS[character]
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "0":
                    continue
                for yy in range(top + gy * scale, top + (gy + 1) * scale):
                    for xx in range(x0 + gx * scale, x0 + (gx + 1) * scale):
                        offset = (yy * width + xx) * 3
                        pixels[offset:offset + 3] = b"\x00\x00\x00"
        x0 += 6 * scale
    rows = [
        b"\x00" + bytes(pixels[y * width * 3:(y + 1) * width * 3])
        for y in range(height)
    ]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + png_chunk(b"IEND", b"")


def vision_request(png: bytes) -> dict:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What color is the top-left quadrant? Reply with exactly RED."},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def ocr_request(png: bytes) -> dict:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Read the text in this image. Reply with exactly GLM53 OCR 8429."},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def canonical_video_fixture() -> bytes:
    return zlib.decompress(base64.b64decode(VIDEO_FIXTURE_ZLIB_B64, validate=True))


def video_rejection_request(gif: bytes) -> dict:
    url = "data:image/gif;base64," + base64.b64encode(gif).decode()
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this video."},
            {"type": "video_url", "video_url": {"url": url}},
        ]}],
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def telemetry_load_request() -> dict:
    return {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": "Output the letter A repeatedly until the token budget is exhausted.",
        }],
        "temperature": 0,
        "max_tokens": 512,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def remote_media_request() -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "http://127.0.0.1:8000/v1/models"}},
        ]}],
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def reasoning_open_stop_request() -> dict:
    return {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": "Compute 12 times 12. If the result is 144, answer with exactly GLM53_REASONING_STOP_OK and nothing else.",
        }],
        "stop": ["144"],
        "temperature": 0,
        "max_tokens": 2300,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def thinking_disabled_stop_request() -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply exactly: BEFORE Question: AFTER"}],
        "stop": ["Question:"],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def tool_request() -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Use the weather tool for Paris, France."}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }}],
        "tool_choice": "auto",
    }


def long_request() -> tuple[dict, str]:
    needle = "NEEDLE_GL53_842917"
    filler = "alpha " * 55_000
    prompt = filler + f"\nHidden retrieval code: {needle}\n" + filler + f"\nReply with exactly {needle}."
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }, prompt


def receipt_route_failures(key: str, row: dict) -> list[str]:
    route = "proxy" if key.startswith("proxy") else "direct"
    base = "http://127.0.0.1:4000" if route == "proxy" else "http://127.0.0.1:8000"
    path = "/v1/models" if key.endswith("models") else "/v1/chat/completions"
    if (
        row.get("requested_url") == base + path
        and row.get("effective_url") == base + path
        and row.get("path") == path
    ):
        return []
    return [f"{key} requested/effective URL/path"]


def response_runtime_failures(key: str, row: object) -> list[str]:
    failures: list[str] = []
    receipt = row if isinstance(row, dict) else {}
    response = receipt.get("response")
    response = response if isinstance(response, dict) else {}
    if response.get("system_fingerprint") != EXPECTED_SYSTEM_FINGERPRINT:
        failures.append(f"{key} system fingerprint")
    started = receipt.get("started_at")
    completed = receipt.get("completed_at")
    created = response.get("created")
    valid_created = (
        isinstance(started, (int, float))
        and not isinstance(started, bool)
        and math.isfinite(started)
        and isinstance(completed, (int, float))
        and not isinstance(completed, bool)
        and math.isfinite(completed)
        and isinstance(created, int)
        and not isinstance(created, bool)
        # vLLM emits integer epoch seconds, so a sub-second request can have a
        # created value floored by less than one second before started_at.
        and math.floor(started) <= created <= math.floor(completed)
    )
    if not valid_created:
        failures.append(f"{key} response created interval")
    return failures


def verify_chat_receipt(key: str, row: dict, expected_request: dict, failures: list[str]) -> None:
    failures.extend(receipt_route_failures(key, row))
    require(row.get("request") == expected_request, f"{key} canonical request", failures)
    require(row.get("request_sha256") == request_sha(expected_request), f"{key} request SHA", failures)
    require(row.get("passed") is True, f"{key} verdict", failures)
    require(row.get("http") == 200, f"{key} HTTP status", failures)
    response = row.get("response", {})
    require(response.get("model") == MODEL, f"{key} response model", failures)
    require(response.get("object") == "chat.completion" and "error" not in response, f"{key} response route/schema", failures)
    failures.extend(response_runtime_failures(key, row))


def remote_media_failures(key: str, row: dict) -> list[str]:
    failures = receipt_route_failures(key, row)
    body = remote_media_request()
    require(row.get("http") == 400, f"{key} HTTP status", failures)
    require(row.get("request") == body, f"{key} canonical request", failures)
    require(row.get("request_sha256") == request_sha(body), f"{key} request SHA", failures)
    require(row.get("passed") is True, f"{key} verdict", failures)
    expected_error = PROXY_REMOTE_MEDIA_ERROR if key.startswith("proxy_") else REMOTE_MEDIA_ERROR
    require(row.get("response") == expected_error, f"{key} exact allowlist error", failures)
    return failures


def video_rejection_failures(key: str, row: dict, fixture: bytes) -> list[str]:
    failures = receipt_route_failures(key, row)
    body = video_rejection_request(fixture)
    require(row.get("http") == 400, f"{key} HTTP status", failures)
    require(row.get("request") == body, f"{key} canonical request", failures)
    require(row.get("request_sha256") == request_sha(body), f"{key} request SHA", failures)
    require(row.get("passed") is True, f"{key} verdict", failures)
    expected_error = PROXY_VIDEO_LIMIT_ERROR if key.startswith("proxy_") else VIDEO_LIMIT_ERROR
    require(row.get("response") == expected_error, f"{key} exact video-zero error", failures)
    return failures


def receipt_interval_failures(acceptance: object, launch_epoch: object) -> list[str]:
    failures: list[str] = []
    if not isinstance(acceptance, dict):
        return ["request receipt cardinality"]
    checks = acceptance.get("checks")
    if not isinstance(checks, dict):
        return ["request receipt cardinality"]
    c4 = checks.get("direct_concurrency_c4")
    results = c4.get("results") if isinstance(c4, dict) else None
    cardinality_ok = (
        set(checks) == EXPECTED_CHECKS
        and isinstance(results, list)
        and len(results) == 4
        and all(isinstance(row, dict) for row in results)
    )
    if not cardinality_ok:
        failures.append("request receipt cardinality")

    rows: list[tuple[str, object]] = [
        (key, checks.get(key))
        for key in sorted(EXPECTED_CHECKS - {"direct_concurrency_c4"})
    ]
    rows.extend(
        (f"direct_concurrency_c4[{index}]", row)
        for index, row in enumerate(results if isinstance(results, list) else [])
    )

    def finite_timestamp(value: object) -> float | None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            return None
        return float(value)

    run_start = finite_timestamp(acceptance.get("started_at"))
    run_end = finite_timestamp(acceptance.get("completed_at"))
    launch = finite_timestamp(launch_epoch)
    bounds_valid = (
        run_start is not None
        and run_end is not None
        and launch is not None
        and launch <= run_start < run_end
    )
    for key, row in rows:
        started = finite_timestamp(row.get("started_at") if isinstance(row, dict) else None)
        completed = finite_timestamp(row.get("completed_at") if isinstance(row, dict) else None)
        valid = (
            bounds_valid
            and started is not None
            and completed is not None
            and run_start is not None
            and run_end is not None
            and launch is not None
            and run_start <= started < completed <= run_end
            and launch <= started
        )
        if not valid:
            failures.append(f"{key} request receipt interval")
    return failures


def expected_runtime_command(
    rank: int, contract: dict[str, object] | None = None
) -> str:
    return derived_runtime_command(contract or reviewed_recipe_command(RECIPE), rank)


def expected_wrapper_command(
    rank: int, contract: dict[str, object] | None = None
) -> str:
    return derived_wrapper_command(contract or reviewed_recipe_command(RECIPE), rank)


CONTAINER_LAUNCHER_COMMAND = (
    "bash -c printf %s c2xlZXAgaW5maW5pdHk= | base64 -d -- | "
    "bash --noprofile --norc"
)
CONTAINER_SHELL_COMMAND = "bash --noprofile --norc"
WATCHDOG_COMMAND = (
    "bash -c SERVE_PID=$(cat /tmp/sparkrun_serve.pid); while kill -0 $SERVE_PID "
    "2>/dev/null; do sleep 5; done; kill 1"
)


def docker_top_inventory_failures(
    text: object,
    rank: int,
    serving_host_pid: int | None,
    serving_container_pid: int | None,
    listener_container_pid: int | None,
    nccl_container_pid: int | None,
    contract: dict[str, object],
) -> list[str]:
    reason = "docker top closed process inventory"
    if rank not in (0, 1) or not isinstance(text, str):
        return [reason]
    lines = text.splitlines()
    if not lines or re.fullmatch(r"PID\s+PPID\s+PGID\s+SID\s+COMMAND", lines[0]) is None:
        return [reason]
    rows: list[dict[str, object]] = []
    pids: set[int] = set()
    for line in lines[1:]:
        match = re.fullmatch(
            r"\s*([1-9]\d*)\s+([0-9]+)\s+([1-9]\d*)\s+([1-9]\d*)\s+(.+)",
            line,
        )
        if match is None:
            return [reason]
        pid = int(match.group(1))
        if pid in pids:
            return [reason]
        pids.add(pid)
        rows.append({
            "pid": pid,
            "ppid": int(match.group(2)),
            "pgid": int(match.group(3)),
            "sid": int(match.group(4)),
            "command": match.group(5),
        })
    exact = {
        CONTAINER_LAUNCHER_COMMAND: 1,
        CONTAINER_SHELL_COMMAND: 1,
        "sleep infinity": 1,
        expected_wrapper_command(rank, contract): 1,
        expected_runtime_command(rank, contract): 1,
        WATCHDOG_COMMAND: 1,
        f"VLLM::Worker_TP{rank}": 1,
    }
    if rank == 0:
        exact["VLLM::EngineCore"] = 1
    resource_tracker = re.compile(
        r"/usr/bin/python3 -c from multiprocessing[.]resource_tracker import main;"
        r"main\([1-9]\d*\)"
    )
    classified: list[str] = []
    resource_count = 0
    transient_sleep_count = 0
    for row in rows:
        command = str(row["command"])
        if command in exact:
            classified.append(command)
        elif resource_tracker.fullmatch(command):
            resource_count += 1
        elif command == "sleep 5":
            transient_sleep_count += 1
        else:
            return [reason]
    if (
        any(classified.count(command) != count for command, count in exact.items())
        or resource_count != 1
        or transient_sleep_count not in (0, 1)
        or len(rows) != sum(exact.values()) + resource_count + transient_sleep_count
    ):
        return [reason]
    by_command = {str(row["command"]): row for row in rows}
    wrapper = by_command[expected_wrapper_command(rank, contract)]
    runtime = by_command[expected_runtime_command(rank, contract)]
    launcher = by_command[CONTAINER_LAUNCHER_COMMAND]
    shell = by_command[CONTAINER_SHELL_COMMAND]
    keepalive = by_command["sleep infinity"]
    watchdog = by_command[WATCHDOG_COMMAND]
    worker = by_command[f"VLLM::Worker_TP{rank}"]
    resource = next(
        row for row in rows if resource_tracker.fullmatch(str(row["command"]))
    )
    transient_sleep = next(
        (row for row in rows if row["command"] == "sleep 5"), None
    )
    shared_shim_pid = wrapper["ppid"]
    owners_match = (
        runtime["pid"] == serving_host_pid
        and serving_container_pid == nccl_container_pid
        and (
            listener_container_pid is None
            if rank == 1
            else serving_container_pid == listener_container_pid
        )
        and shared_shim_pid not in pids
        and launcher["ppid"] == shared_shim_pid
        and watchdog["ppid"] == shared_shim_pid
        and launcher["pgid"] == launcher["pid"]
        and launcher["sid"] == launcher["pid"]
        and shell["ppid"] == launcher["pid"]
        and shell["pgid"] == launcher["pid"]
        and shell["sid"] == launcher["pid"]
        and keepalive["ppid"] == shell["pid"]
        and keepalive["pgid"] == launcher["pid"]
        and keepalive["sid"] == launcher["pid"]
        and wrapper["pgid"] == wrapper["pid"]
        and wrapper["sid"] == wrapper["pid"]
        and runtime["ppid"] == wrapper["pid"]
        and runtime["pgid"] == runtime["pid"]
        and runtime["sid"] == runtime["pid"]
        and watchdog["pgid"] == watchdog["pid"]
        and watchdog["sid"] == watchdog["pid"]
        and resource["ppid"] == runtime["pid"]
        and resource["pgid"] == runtime["pid"]
        and resource["sid"] == runtime["pid"]
        and (
            transient_sleep is None
            or (
                transient_sleep["ppid"] == watchdog["pid"]
                and transient_sleep["pgid"] == watchdog["pid"]
                and transient_sleep["sid"] == watchdog["pid"]
            )
        )
    )
    if rank == 0:
        engine = by_command["VLLM::EngineCore"]
        owners_match = owners_match and (
            engine["ppid"] == runtime["pid"]
            and engine["pgid"] == runtime["pid"]
            and engine["sid"] == runtime["pid"]
            and worker["ppid"] == engine["pid"]
            and worker["pgid"] == runtime["pid"]
            and worker["sid"] == runtime["pid"]
        )
    else:
        owners_match = owners_match and (
            worker["ppid"] == runtime["pid"]
            and worker["pgid"] == runtime["pid"]
            and worker["sid"] == runtime["pid"]
        )
    if not owners_match:
        return ["docker top process identity join"]
    return []


def docker_top_runtime_host_pid(
    text: object, rank: int, contract: dict[str, object]
) -> int | None:
    if not isinstance(text, str):
        return None
    expected = expected_runtime_command(rank, contract)
    matches: list[int] = []
    for line in text.splitlines()[1:]:
        match = re.fullmatch(
            r"\s*([1-9]\d*)\s+[0-9]+\s+[1-9]\d*\s+[1-9]\d*\s+(.+)", line
        )
        if match is not None and match.group(2) == expected:
            matches.append(int(match.group(1)))
    return matches[0] if len(matches) == 1 else None


def docker_top_wrapper_host_pid(
    text: object, rank: int, contract: dict[str, object]
) -> int | None:
    if not isinstance(text, str):
        return None
    expected = expected_wrapper_command(rank, contract)
    matches: list[int] = []
    for line in text.splitlines()[1:]:
        match = re.fullmatch(
            r"\s*([1-9]\d*)\s+[0-9]+\s+[1-9]\d*\s+[1-9]\d*\s+(.+)", line
        )
        if match is not None and match.group(2) == expected:
            matches.append(int(match.group(1)))
    return matches[0] if len(matches) == 1 else None


def docker_top_wrapper_parent_host_pid(
    text: object, rank: int, contract: dict[str, object]
) -> int | None:
    if not isinstance(text, str):
        return None
    expected = expected_wrapper_command(rank, contract)
    matches: list[int] = []
    for line in text.splitlines()[1:]:
        match = re.fullmatch(
            r"\s*[1-9]\d*\s+([1-9]\d*)\s+[1-9]\d*\s+[1-9]\d*\s+(.+)",
            line,
        )
        if match is not None and match.group(2) == expected:
            matches.append(int(match.group(1)))
    return matches[0] if len(matches) == 1 else None


def worker_listener_failures(
    receipt: object, host: str, container: str
) -> list[str]:
    reason = "worker negative listener receipt"
    if not isinstance(receipt, dict):
        return [reason]
    try:
        payload = json.loads(receipt.get("stdout", ""))
    except (json.JSONDecodeError, TypeError):
        return [reason]
    expected_payload = {
        "namespace": "host",
        "host": host,
        "container": container,
        "rank": 1,
        "endpoint": {"port": 8000},
        "listeners": [],
    }
    valid = (
        receipt.get("argv")
        == expected_remote_argv(host, direct_listener_command(container, host, 1))
        and receipt.get("returncode") == 0
        and receipt.get("stderr") == ""
        and payload == expected_payload
    )
    return [] if valid else [reason]


def runtime_overlay_command(container: str) -> str:
    return (
        f"docker exec {shlex.quote(container)} bash -euo pipefail -c "
        + shlex.quote(
            "python3 /workspace/mods/glm-5.3-flash-exl3-upstream-850k/"
            "verify_runtime_patch_state.py && "
            "python3 /workspace/mods/glm-5.3-flash-exl3-upstream-850k/"
            "test_suppress_stops_multitoken.py --production && "
            "python3 - <<'PY'\n"
            "import exllamav3_ext as e\n"
            "required=('exl3_moe','exl3_fat_gemm','exl3_fat_gemm_scatter',"
            "'exl3_fat_moe_gateup','exl3_fat_moe_down','exl3_fat_moe_gather')\n"
            "print('symbols=' + ','.join(x for x in required if hasattr(e,x)))\n"
            "assert all(hasattr(e,x) for x in required)\n"
            "PY"
        )
    )


def expected_remote_argv(host: str, command: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", host, "bash", "-lc", shlex.quote(command)]


def serving_process_command(container: str) -> str:
    return (
        f"docker exec {shlex.quote(container)} bash -lc "
        + shlex.quote(
            "pid=$(pgrep -fo '/usr/local/bin/vllm serve'); test -n \"$pid\"; "
            "cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline); "
            "printf 'pid=%s\\ncommand=%s\\n' \"$pid\" \"${cmd% }\""
        )
    )


def nccl_loaded_command(container: str, serving_pid: str) -> str:
    return (
        f"docker exec {shlex.quote(container)} bash -lc "
        + shlex.quote(
            f"pid={serving_pid}; test -r /proc/{serving_pid}/maps; "
            f"lib=$(awk '/libnccl[.]so[.]2/{{print $6; exit}}' /proc/{serving_pid}/maps); "
            "test -n \"$lib\"; printf 'pid=%s\\npath=%s\\nsha256=' \"$pid\" "
            "\"$(readlink -f \"$lib\")\"; sha256sum \"$lib\" | cut -d' ' -f1; "
            "python3 -c \"import importlib.metadata as m; "
            "print('package_version='+m.version('nvidia-nccl-cu13'))\""
        )
    )


RDMA_COMMAND = (
    "rdma link; for d in /sys/class/infiniband/*; do h=$(basename \"$d\"); "
    "printf '%s rate=' \"$h\"; cat \"$d/ports/1/rate\"; "
    "printf '%s state=' \"$h\"; cat \"$d/ports/1/state\"; done"
)
MEMORY_COMMAND = (
    "grep -E '^(MemFree|MemAvailable|SwapFree):' /proc/meminfo; "
    "awk '/oom_kill /{print \"oom_kill=\"$2}' /proc/vmstat"
)
SAFE_ENV_NAMES = (
    "ABLIT",
    "EXL3_FAT_GROUPED",
    "GLM53_ADAPTIVE_K",
    "GLM53_DENSE_FP8",
    "GLM53_INDEXER_WORKSPACE",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
)


def runtime_env_command() -> str:
    script = (
        "import os\n"
        f"names={SAFE_ENV_NAMES!r}\n"
        "for name in names: print(f'{name}={os.environ[name]}')"
    )
    return f"python3 -c {shlex.quote(script)}"


def runtime_mod_manifest_command(container: str, expected_digest: str) -> str:
    inner = (
        "cd /workspace/mods/glm-5.3-flash-exl3-upstream-850k; "
        "actual=$(sha256sum SHA256SUMS | cut -d' ' -f1); "
        f"test \"$actual\" = {shlex.quote(expected_digest)}; "
        "printf 'manifest_sha256=%s\\n' \"$actual\"; "
        "sha256sum -c SHA256SUMS"
    )
    return f"docker exec {shlex.quote(container)} bash -euo pipefail -c {shlex.quote(inner)}"


def mod_manifest_failures(mod: Path) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    manifest = mod / "SHA256SUMS"
    if not manifest.is_file() or sha(manifest) != EXPECTED_MOD_MANIFEST_SHA256:
        return ["reviewed local mod manifest digest"], []
    entries: list[str] = []
    expected_hashes: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/].*)", line)
        if match is None or match.group(2) in entries:
            failures.append("reviewed local mod manifest grammar")
            continue
        entries.append(match.group(2))
        expected_hashes[match.group(2)] = match.group(1)
    actual = {
        str(path.relative_to(mod))
        for path in mod.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and "__pycache__" not in path.parts
    }
    if set(entries) != actual or len(entries) != len(actual):
        failures.append("reviewed local mod manifest completeness")
    elif any(sha(mod / entry) != expected_hashes[entry] for entry in entries):
        failures.append("reviewed local mod file hashes")
    return failures, entries
EXPECTED_RDMA = {
    "rocep1s0f0": {"rate": "40 Gb/sec (4X QDR)", "state": "1: DOWN"},
    "rocep1s0f1": {"rate": "200 Gb/sec (4X HDR)", "state": "4: ACTIVE"},
    "roceP2p1s0f0": {"rate": "40 Gb/sec (4X QDR)", "state": "1: DOWN"},
    "roceP2p1s0f1": {"rate": "200 Gb/sec (4X HDR)", "state": "4: ACTIVE"},
}
EXPECTED_SOCKET_IFNAMES = ("enP7s7", "enp1s0f1np1", "enP2p1s0f1np1")
EXPECTED_ROCE = {
    "192.168.178.47": [
        {"hca": "rocep1s0f1", "gid_index": 3, "gid": "::ffff:192.168.3.72", "gid_type": "RoCE v2", "netdev": "enp1s0f1np1", "ipv4": "192.168.3.72"},
        {"hca": "roceP2p1s0f1", "gid_index": 3, "gid": "::ffff:192.168.2.72", "gid_type": "RoCE v2", "netdev": "enP2p1s0f1np1", "ipv4": "192.168.2.72"},
    ],
    "192.168.178.46": [
        {"hca": "rocep1s0f1", "gid_index": 3, "gid": "::ffff:192.168.3.183", "gid_type": "RoCE v2", "netdev": "enp1s0f1np1", "ipv4": "192.168.3.183"},
        {"hca": "roceP2p1s0f1", "gid_index": 3, "gid": "::ffff:192.168.2.183", "gid_type": "RoCE v2", "netdev": "enP2p1s0f1np1", "ipv4": "192.168.2.183"},
    ],
}


def expected_host_commands(
    host: str,
    container: str,
    serving_pid: str,
    launch_epoch: int,
    acceptance_completed: float,
    readiness_epoch: int,
) -> dict[str, list[str]]:
    qcontainer = shlex.quote(container)
    commands = {
        "container_discovery": (
            f"docker ps --filter name={shlex.quote(CLUSTER)} --format '{{{{.Names}}}}'"
        ),
        "earlyoom": "systemctl is-active earlyoom || true",
        "docker_inspect": f"docker inspect {qcontainer}",
        "image_inspect": f"docker image inspect {shlex.quote(IMAGE)}",
        "docker_top": f"docker top {qcontainer} -eo pid,ppid,pgid,sid,args",
        "docker_logs": f"docker logs {qcontainer}",
        "serve_log": f"docker exec {qcontainer} cat /tmp/sparkrun_serve.log",
        "runtime_env": f"docker exec {qcontainer} bash -lc " + shlex.quote(runtime_env_command()),
        "roce_records": roce_records_command(container),
        "runtime_overlay": runtime_overlay_command(container),
        "runtime_mod_manifest": runtime_mod_manifest_command(
            container, EXPECTED_MOD_MANIFEST_SHA256
        ),
        "serving_process": serving_process_command(container),
        "nccl_loaded": nccl_loaded_command(container, serving_pid),
        "rdma": RDMA_COMMAND,
        "memory": MEMORY_COMMAND,
        "kernel_since_launch": f"journalctl -k --since '@{launch_epoch}' --no-pager",
        "kernel_readiness_to_acceptance": (
            f"journalctl -k --since '@{readiness_epoch}' "
            f"--until '@{acceptance_completed + 1.0:.6f}' --no-pager"
        ),
        "kernel_after_acceptance": (
            f"journalctl -k --since '@{int(acceptance_completed)}' --no-pager"
        ),
    }
    return {key: expected_remote_argv(host, command) for key, command in commands.items()}


def parse_rdma(text: str) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"(\S+) (rate|state)=(.+)", line)
        if match:
            hca, key, value = match.groups()
            parsed.setdefault(hca, {})[key] = value
    return parsed


def roce_binding_failures(
    host: str,
    env_text: str,
    records: object,
    telemetry_hcas: set[str],
) -> list[str]:
    failures: list[str] = []
    env: dict[str, str] = {}
    valid_env = True
    for line in env_text.splitlines():
        if "=" not in line:
            valid_env = False
            continue
        key, value = line.split("=", 1)
        if key in env:
            valid_env = False
        env[key] = value
    expected_records = EXPECTED_ROCE[host]
    expected_hcas = tuple(row["hca"] for row in expected_records)
    expected_netdevs = {row["netdev"] for row in expected_records}
    selected_hcas = tuple(env.get("NCCL_IB_HCA", "").split(","))
    if not valid_env or selected_hcas != expected_hcas:
        failures.append("runtime NCCL_IB_HCA")
    if env.get("NCCL_IB_GID_INDEX") != "3":
        failures.append("runtime NCCL_IB_GID_INDEX")
    socket_ifnames = tuple(env.get("NCCL_SOCKET_IFNAME", "").split(","))
    if socket_ifnames != EXPECTED_SOCKET_IFNAMES or not expected_netdevs.issubset(socket_ifnames):
        failures.append("runtime NCCL_SOCKET_IFNAME")
    semantic_records = isinstance(records, list) and len(records) == len(expected_records)
    if isinstance(records, list) and semantic_records:
        for row, expected in zip(records, expected_records):
            try:
                semantic_records = semantic_records and set(row) == set(expected)
                semantic_records = semantic_records and all(
                    row[key] == expected[key] for key in expected if key != "gid"
                )
                gid = ipaddress.IPv6Address(str(row["gid"]))
                expected_gid = ipaddress.IPv6Address(str(expected["gid"]))
                ipv4 = ipaddress.IPv4Address(str(row["ipv4"]))
                semantic_records = semantic_records and gid == expected_gid
                semantic_records = semantic_records and gid.ipv4_mapped == ipv4
                semantic_records = semantic_records and row["gid_type"] == "RoCE v2"
                semantic_records = semantic_records and row["gid_index"] == 3
            except (KeyError, TypeError, ValueError):
                semantic_records = False
    if not semantic_records:
        failures.append("runtime RoCE records")
    if telemetry_hcas != set(expected_hcas) or telemetry_hcas != set(selected_hcas):
        failures.append("telemetry HCA names")
    return failures


LISTENER_PROBE = r'''import json,os,sys
port=int(sys.argv[1]); sockets=[]
for table,family in (("/proc/net/tcp","ipv4"),("/proc/net/tcp6","ipv6")):
    try: lines=open(table).read().splitlines()[1:]
    except OSError: continue
    for line in lines:
        fields=line.split(); local,state,inode=fields[1],fields[3],fields[9]
        address,hexport=local.split(":")
        if int(hexport,16)==port and state=="0A":
            bind="0.0.0.0" if family=="ipv4" and int(address,16)==0 else ("::" if int(address,16)==0 else address)
            sockets.append((inode,bind))
assert len(sockets)==1,sockets
inode,bind=sockets[0]; owners=[]
for name in os.listdir("/proc"):
    if not name.isdigit(): continue
    try:
        if any(os.readlink("/proc/"+name+"/fd/"+fd)=="socket:["+inode+"]" for fd in os.listdir("/proc/"+name+"/fd")):
            command=open("/proc/"+name+"/cmdline","rb").read().replace(b"\0",b" ").decode().strip()
            owners.append((int(name),command))
    except (OSError,PermissionError): pass
assert len(owners)==1,owners
pid,command=owners[0]
print(json.dumps({"bind":bind,"port":port,"inode":inode,"pid":pid,"command":command},sort_keys=True))'''


DIRECT_LISTENER_PROBE = r'''import json,os,sys
namespace,host,container,rank,port=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]); sockets=[]
for table,family in (("/proc/net/tcp","ipv4"),("/proc/net/tcp6","ipv6")):
    try: lines=open(table).read().splitlines()[1:]
    except OSError: continue
    for line in lines:
        fields=line.split(); local,state,inode=fields[1],fields[3],fields[9]
        address,hexport=local.split(":")
        if int(hexport,16)==port and state=="0A":
            bind="0.0.0.0" if family=="ipv4" and int(address,16)==0 else ("::" if int(address,16)==0 else address)
            sockets.append((inode,bind))
listeners=[]
for inode,bind in sockets:
    owners=[]
    for name in os.listdir("/proc"):
        if not name.isdigit(): continue
        try:
            if any(os.readlink("/proc/"+name+"/fd/"+fd)=="socket:["+inode+"]" for fd in os.listdir("/proc/"+name+"/fd")):
                command=open("/proc/"+name+"/cmdline","rb").read().replace(b"\0",b" ").decode().strip()
                owners.append((int(name),command))
        except (OSError,PermissionError): pass
    assert len(owners)==1,owners
    pid,command=owners[0]
    listeners.append({"bind":bind,"inode":inode,"pid":pid,"command":command})
print(json.dumps({"namespace":namespace,"host":host,"container":container,"rank":rank,"endpoint":{"port":port},"listeners":listeners},sort_keys=True))'''


def direct_listener_command(container: str, host: str, rank: int) -> str:
    args = f"{shlex.quote(host)} {shlex.quote(container)} {rank} 8000"
    if rank == 0:
        return (
            f"docker exec {shlex.quote(container)} /usr/bin/python3 -S -c "
            f"{shlex.quote(DIRECT_LISTENER_PROBE)} container {args}"
        )
    if rank == 1:
        return (
            f"/usr/bin/python3 -S -c {shlex.quote(DIRECT_LISTENER_PROBE)} "
            f"host {args}"
        )
    raise ValueError("rank must be 0 or 1")


REQUIRED_SERVE_MARKERS = (
    "configured_tier=grouped effective_tier=grouped",
    "Using fp8_ds_mla KV cache format",
    "[glm53-indexer-workspace] rightsize",
    "Graph capturing finished",
    "DFlash2 drafter KV",
)
FATAL_SERVE_MARKERS = (
    "Traceback (most recent call last):",
    "NVRM: Xid",
    "CUDA error: an illegal memory access",
)
SERVE_MARKER_PROBE = r'''import json,pathlib,re
text=pathlib.Path("/tmp/sparkrun_serve.log").read_text()
required=("configured_tier=grouped effective_tier=grouped","Using fp8_ds_mla KV cache format","[glm53-indexer-workspace] rightsize","Graph capturing finished","DFlash2 drafter KV")
fatal=("Traceback (most recent call last):","NVRM: Xid","CUDA error: an illegal memory access")
counts={marker:text.count(marker) for marker in required}
matches=re.findall(r"\[glm53-e3-executed\] grouped_calls=([1-9][0-9]*) fat_expert_runs=([1-9][0-9]*) configured_tier=(\S+) effective_tier=(\S+)",text)
e3=[{"grouped_calls":int(a),"fat_expert_runs":int(b),"configured_tier":c,"effective_tier":d} for a,b,c,d in matches]
found=[marker for marker in fatal if marker in text]
assert all(value>0 for value in counts.values()) and len(e3)==1 and e3[0]["configured_tier"]=="grouped" and e3[0]["effective_tier"]=="grouped" and not found
print(json.dumps({"required_marker_counts":counts,"e3_markers":e3,"fatal_matches":found},sort_keys=True))'''


def serve_marker_command(container: str) -> str:
    return (
        f"docker exec {shlex.quote(container)} /usr/bin/python3 -S -c "
        f"{shlex.quote(SERVE_MARKER_PROBE)}"
    )


def serve_marker_failures(receipt: object, host: str, container: str) -> list[str]:
    reason = "structured E3 runtime execution marker and fatal scan"
    if not isinstance(receipt, dict):
        return [reason]
    try:
        payload = json.loads(receipt.get("stdout", ""))
    except (TypeError, json.JSONDecodeError):
        return [reason]
    counts = payload.get("required_marker_counts") if isinstance(payload, dict) else None
    e3 = payload.get("e3_markers") if isinstance(payload, dict) else None
    valid_e3 = (
        isinstance(e3, list)
        and len(e3) == 1
        and isinstance(e3[0], dict)
        and isinstance(e3[0].get("grouped_calls"), int)
        and e3[0].get("grouped_calls", 0) > 0
        and isinstance(e3[0].get("fat_expert_runs"), int)
        and e3[0].get("fat_expert_runs", 0) > 0
        and e3[0].get("configured_tier") == "grouped"
        and e3[0].get("effective_tier") == "grouped"
        and set(e3[0]) == {
            "grouped_calls", "fat_expert_runs", "configured_tier", "effective_tier"
        }
    )
    valid = (
        receipt.get("argv") == expected_remote_argv(host, serve_marker_command(container))
        and receipt.get("returncode") == 0
        and receipt.get("stderr") == ""
        and isinstance(payload, dict)
        and set(payload) == {"required_marker_counts", "e3_markers", "fatal_matches"}
        and isinstance(counts, dict)
        and set(counts) == set(REQUIRED_SERVE_MARKERS)
        and all(isinstance(counts.get(marker), int) and counts[marker] > 0 for marker in REQUIRED_SERVE_MARKERS)
        and valid_e3
        and payload.get("fatal_matches") == []
    )
    return [] if valid else [reason]


PID_NAMESPACE_PROBE = r'''import pathlib,re,sys
host_pid=int(sys.argv[1]); text=pathlib.Path(f"/proc/{host_pid}/status").read_text()
match=re.search(r"^NSpid:\s+([0-9\s]+)$",text,re.MULTILINE); assert match
pids=[int(value) for value in match.group(1).split()]; assert len(pids)==2 and pids[0]==host_pid
print(f"host_pid={host_pid}\ncontainer_pid={pids[1]}")'''


def pid_namespace_command(host_pid: int) -> str:
    if host_pid <= 0:
        raise ValueError("host PID must be positive")
    return f"python3 -c {shlex.quote(PID_NAMESPACE_PROBE)} {host_pid}"


def pid_namespace_failures(
    receipt: object, host: str, host_pid: int, container_pid: int
) -> list[str]:
    reason = "serving PID namespace join"
    if not isinstance(receipt, dict):
        return [reason]
    valid = (
        receipt.get("argv")
        == expected_remote_argv(host, pid_namespace_command(host_pid))
        and receipt.get("returncode") == 0
        and receipt.get("stderr") == ""
        and receipt.get("stdout")
        == f"host_pid={host_pid}\ncontainer_pid={container_pid}\n"
    )
    return [] if valid else [reason]


WRAPPER_LINEAGE_PROBE = r'''import json,pathlib,re,sys
wrapper_host_pid=int(sys.argv[1]); status=pathlib.Path(f"/proc/{wrapper_host_pid}/status").read_text()
parent_match=re.search(r"^PPid:\s+([1-9][0-9]*)$",status,re.MULTILINE); assert parent_match
nspid_match=re.search(r"^NSpid:\s+([0-9\s]+)$",status,re.MULTILINE); assert nspid_match
nspids=[int(value) for value in nspid_match.group(1).split()]; assert len(nspids)==2 and nspids[0]==wrapper_host_pid
parent_host_pid=int(parent_match.group(1)); argv=pathlib.Path(f"/proc/{parent_host_pid}/cmdline").read_bytes().split(b"\0"); argv=[item.decode() for item in argv if item]
assert len(argv)==7 and argv[0]=="/usr/bin/containerd-shim-runc-v2" and argv[1:4]==["-namespace","moby","-id"] and re.fullmatch(r"[0-9a-f]{64}",argv[4]) and argv[5:]==["-address","/run/containerd/containerd.sock"]
payload={"wrapper_host_pid":wrapper_host_pid,"parent_host_pid":parent_host_pid,"parent_executable":argv[0],"parent_namespace":argv[2],"parent_container_id":argv[4],"parent_address":argv[6]}
print(json.dumps(payload,sort_keys=True))'''


def wrapper_lineage_command(wrapper_host_pid: int) -> str:
    if wrapper_host_pid <= 0:
        raise ValueError("wrapper host PID must be positive")
    return f"python3 -c {shlex.quote(WRAPPER_LINEAGE_PROBE)} {wrapper_host_pid}"


def wrapper_lineage_failures(
    receipt: object,
    host: str,
    wrapper_host_pid: int,
    parent_host_pid: int,
    container_id: str,
) -> list[str]:
    reason = "wrapper shim and PID namespace join"
    if not isinstance(receipt, dict):
        return [reason]
    try:
        payload = json.loads(receipt.get("stdout", ""))
    except (TypeError, json.JSONDecodeError):
        return [reason]
    expected_keys = {
        "wrapper_host_pid", "parent_host_pid",
        "parent_executable", "parent_namespace", "parent_container_id",
        "parent_address",
    }
    valid = (
        receipt.get("argv")
        == expected_remote_argv(host, wrapper_lineage_command(wrapper_host_pid))
        and receipt.get("returncode") == 0
        and receipt.get("stderr") == ""
        and isinstance(payload, dict)
        and set(payload) == expected_keys
        and payload.get("wrapper_host_pid") == wrapper_host_pid
        and payload.get("parent_host_pid") == parent_host_pid
        and payload.get("parent_executable") == "/usr/bin/containerd-shim-runc-v2"
        and payload.get("parent_namespace") == "moby"
        and payload.get("parent_container_id") == container_id
        and re.fullmatch(r"[0-9a-f]{64}", container_id) is not None
        and payload.get("parent_address") == "/run/containerd/containerd.sock"
    )
    return [] if valid else [reason]


def roce_records_command(container: str) -> str:
    inner = (
        "IFS=, read -r -a hcas <<< \"${NCCL_IB_HCA:?NCCL_IB_HCA is required}\"; "
        "python3 /workspace/mods/glm-5.3-flash-exl3-upstream-850k/verify_roce_gid.py "
        "--json --gid-index \"${NCCL_IB_GID_INDEX:?NCCL_IB_GID_INDEX is required}\" "
        "\"${hcas[@]}\""
    )
    return f"docker exec {shlex.quote(container)} bash -euo pipefail -c {shlex.quote(inner)}"


def proxy_parent_argv(archive: str) -> list[str]:
    return [
        "/home/max/.local/bin/uv",
        "tool",
        "uvx",
        "--from",
        "litellm[proxy]==1.82.6",
        "litellm",
        "--config",
        "/home/max/.cache/sparkrun/proxy/litellm_config.yaml",
        "--host",
        "0.0.0.0",
        "--port",
        "4000",
    ]


def proxy_supervisor_argv() -> list[str]:
    return [
        "/home/max/.local/share/uv/tools/sparkrun/bin/python",
        "-m",
        "sparkrun.proxy.autodiscover",
        "/home/max/.cache/sparkrun/proxy/autodiscover.yaml",
    ]


def proxy_child_argv(archive: str) -> list[str]:
    base = f"/home/max/.cache/uv/archive-v0/{archive}/bin"
    return [
        f"{base}/python",
        f"{base}/litellm",
        "--config",
        "/home/max/.cache/sparkrun/proxy/litellm_config.yaml",
        "--host",
        "0.0.0.0",
        "--port",
        "4000",
    ]


def canonical_proxy_status(pid: int, supervisor_pid: int) -> str:
    return (
        "Proxy status: running\n"
        f"  PID:     {pid}\n"
        "  Host:    0.0.0.0\n"
        "  Port:    4000\n"
        f"  Auto-discover: running (PID {supervisor_pid})\n"
        "Registered models (1):\n"
        "  GLM-5.3-Flash-EXL3\n"
        "    -> http://192.168.178.47:8000/v1\n"
    )


def proxy_status_pid(text: object) -> int | None:
    if not isinstance(text, str) or not proxy_status_valid(text):
        return None
    matches = re.findall(r"^\s*PID:\s*([1-9]\d*)\s*$", text, re.MULTILINE)
    return int(matches[0]) if len(matches) == 1 else None


def proxy_status_supervisor_pid(text: object) -> int | None:
    if not isinstance(text, str) or not proxy_status_valid(text):
        return None
    matches = re.findall(
        r"^\s*Auto-discover:\s*running \(PID ([1-9]\d*)\)\s*$",
        text,
        re.MULTILINE,
    )
    return int(matches[0]) if len(matches) == 1 else None


def proxy_process_role(row: object) -> str | None:
    if not isinstance(row, dict) or not isinstance(row.get("argv"), list):
        return None
    argv = row["argv"]
    if argv == proxy_supervisor_argv():
        return "autodiscover_supervisor"
    if argv == proxy_parent_argv("ignored"):
        return "uv_parent"
    if len(argv) == 8:
        match = re.fullmatch(
            r"/home/max/[.]cache/uv/archive-v0/([^/]+)/bin/python", str(argv[0])
        )
        if match and argv == proxy_child_argv(match.group(1)):
            return "listener_child"
    return None


def proxy_binding_failures(runtime: object) -> list[str]:
    failures: list[str] = []
    if not isinstance(runtime, dict):
        return ["proxy process identity"]
    status = runtime.get("proxy_status", {})
    status_text = status.get("stdout") if isinstance(status, dict) else None
    status_pid = proxy_status_pid(status_text)
    status_supervisor_pid = proxy_status_supervisor_pid(status_text)
    processes = runtime.get("proxy_processes")
    valid_processes = (
        isinstance(processes, list)
        and len(processes) == 3
        and all(isinstance(row, dict) for row in processes)
    )
    by_role = {
        row.get("role"): row
        for row in processes or []
        if isinstance(row, dict) and proxy_process_role(row) == row.get("role")
    }
    supervisor = by_role.get("autodiscover_supervisor", {})
    parent = by_role.get("uv_parent", {})
    child = by_role.get("listener_child", {})
    identity_fields = ("pid", "ppid", "pgid", "sid", "started_at")
    fields_valid = all(
        isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
        for row in (supervisor, parent, child)
        for field in identity_fields
    )
    lineage_valid = (
        valid_processes
        and len(by_role) == 3
        and fields_valid
        and status_pid == parent.get("pid")
        and status_supervisor_pid == supervisor.get("pid")
        and supervisor.get("pgid") == supervisor.get("pid")
        and supervisor.get("sid") == supervisor.get("pid")
        and parent.get("ppid") == supervisor.get("pid")
        and parent.get("pgid") == parent.get("pid")
        and parent.get("sid") == parent.get("pid")
        and child.get("ppid") == parent.get("pid")
        and child.get("pgid") == parent.get("pgid")
        and child.get("sid") == parent.get("sid")
        and 0 < supervisor.get("started_at", 0) <= parent.get("started_at", 0)
        and parent.get("started_at", 0) <= child.get("started_at", 0)
    )
    if not lineage_valid:
        failures.append("proxy process identity")

    listener = runtime.get("proxy_listener")
    listener_row = listener if isinstance(listener, dict) else {}
    listener_valid = bool(listener_row) and all(
        listener_row.get(field) == child.get(field)
        for field in (*identity_fields, "role", "argv")
    )
    listener_valid = (
        listener_valid
        and listener_row.get("bind") == "0.0.0.0"
        and listener_row.get("port") == 4000
        and bool(re.fullmatch(r"[1-9]\d*", str(listener_row.get("inode", ""))))
    )
    if not listener_valid:
        failures.append("proxy listener identity")

    sockets = runtime.get("proxy_upstream_sockets")
    socket_valid = isinstance(sockets, list) and len(sockets) >= 1
    if isinstance(sockets, list):
        socket_valid = socket_valid and all(
            isinstance(row, dict)
            and row.get("owner_pid") == child.get("pid")
            and row.get("state") == "ESTABLISHED"
            and row.get("remote_address") == "192.168.178.47"
            and row.get("remote_port") == 8000
            and bool(re.fullmatch(r"[1-9]\d*", str(row.get("inode", ""))))
            for row in sockets
        )
    if not socket_valid:
        failures.append("proxy established upstream sockets")
    return failures


def proxy_command_valid(command: object) -> bool:
    if not isinstance(command, str):
        return False
    pattern = (
        r"/home/max/[.]cache/uv/archive-v0/([^/]+)/bin/python "
        r"/home/max/[.]cache/uv/archive-v0/\1/bin/litellm "
        r"--config /home/max/[.]cache/sparkrun/proxy/litellm_config[.]yaml "
        r"--host 0[.]0[.]0[.]0 --port 4000"
    )
    return re.fullmatch(pattern, command) is not None


def proxy_status_valid(text: str) -> bool:
    required = (
        "Proxy status: running",
        "Host:    0.0.0.0",
        "Port:    4000",
        "Registered models (1):",
        "  GLM-5.3-Flash-EXL3",
        "    -> http://192.168.178.47:8000/v1",
    )
    return all(text.count(item) == 1 for item in required)


def expected_runtime_overlay_argv(host: str, container: str) -> list[str]:
    return expected_remote_argv(host, runtime_overlay_command(container))


def expected_postready_argv(host: str, container: str, filename: str) -> list[str]:
    return expected_remote_argv(
        host, f"docker exec {shlex.quote(container)} cat /tmp/{filename}"
    )


def parse_telemetry(path: Path) -> dict:
    lines = path.read_text().splitlines()
    index = 0

    def take(pattern: str, label: str) -> re.Match[str]:
        nonlocal index
        if index >= len(lines):
            raise ValueError(f"incomplete telemetry: expected {label}")
        match = re.fullmatch(pattern, lines[index])
        if match is None:
            raise ValueError(
                f"telemetry grammar error at line {index + 1}: expected {label}"
            )
        index += 1
        return match

    def iso_time(value: str) -> float:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("telemetry timestamps must include a timezone")
        return parsed.timestamp()

    number = r"(-?(?:\d+(?:[.]\d*)?|[.]\d+)(?:[eE][+-]?\d+)?)"
    labels = rf'engine="0",model_name="{re.escape(MODEL)}"'
    metric_patterns = (
        ("running", rf"vllm:num_requests_running\{{{labels}\}} {number}"),
        ("waiting", rf"vllm:num_requests_waiting\{{{labels}\}} {number}"),
        (
            "waiting_capacity",
            rf'vllm:num_requests_waiting_by_reason\{{{labels},reason="capacity"\}} {number}',
        ),
        (
            "waiting_deferred",
            rf'vllm:num_requests_waiting_by_reason\{{{labels},reason="deferred"\}} {number}',
        ),
        ("kv_cache_usage", rf"vllm:kv_cache_usage_perc\{{{labels}\}} {number}"),
    )
    samples: dict[str, dict] = {}
    for sample_name in ("before", "during", "after"):
        sample_match = take(
            rf"sample={sample_name} time=(\S+)", f"{sample_name} sample header"
        )
        sample: dict = {
            "timestamp": iso_time(sample_match.group(1)),
            "hosts": {},
            "metrics": {},
        }
        for metric_name, pattern in metric_patterns:
            metric = take(pattern, metric_name)
            sample["metrics"][metric_name] = float(metric.group(1))
        sample["running"] = sample["metrics"]["running"]

        for expected_host in EXPECTED_HOST_ORDER:
            started = take(
                rf"host={re.escape(expected_host)} acquisition_started_at=(\S+)",
                f"{expected_host} acquisition start",
            )
            host_row = {
                "acquisition_started_at": iso_time(started.group(1)),
                "gpu_samples": [],
                "hcas": {},
            }
            for gpu_index in range(10):
                gpu = take(
                    r"(P\d+), (\d+) %, (\d+) MHz, ([0-9.]+) W",
                    f"{expected_host} GPU sample {gpu_index + 1}/10",
                )
                host_row["gpu_samples"].append({
                    "pstate": gpu.group(1),
                    "utilization": int(gpu.group(2)),
                    "clock_mhz": int(gpu.group(3)),
                    "power_w": float(gpu.group(4)),
                })
            for hca in ("rocep1s0f1", "roceP2p1s0f1"):
                hca_row: dict[str, object] = {}
                for key in ("rate", "state", "xmit", "rcv"):
                    value = take(
                        rf"{re.escape(hca)} {key}=(.+)",
                        f"{expected_host} {hca} {key}",
                    ).group(1)
                    hca_row[key] = int(value) if key in {"xmit", "rcv"} else value
                host_row["hcas"][hca] = hca_row
            completed = take(
                rf"host={re.escape(expected_host)} acquisition_completed_at=(\S+)",
                f"{expected_host} acquisition completion",
            )
            host_row["acquisition_completed_at"] = iso_time(completed.group(1))
            sample["hosts"][expected_host] = host_row
        samples[sample_name] = sample
    if index != len(lines):
        raise ValueError(f"unknown telemetry line {index + 1}")
    return samples


def readme_runtime_completion_failures(root: Path, runtime: dict) -> list[str]:
    try:
        readme = (root / "README.md").read_text()
        match = re.search(
            r"^- Runtime capture completed: `([^`]+)`$", readme, re.MULTILINE
        )
        completed = runtime.get("completed_at")
        if (
            match is None
            or not isinstance(completed, (int, float))
            or isinstance(completed, bool)
            or not math.isfinite(completed)
            or round(datetime.fromisoformat(match.group(1)).timestamp(), 6)
            != round(completed, 6)
        ):
            return ["README runtime capture completion"]
    except (OSError, TypeError, ValueError):
        return ["README runtime capture completion"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--mod-dir", type=Path, default=MOD)
    args = parser.parse_args()
    root = args.root.resolve()
    recipe = args.recipe.resolve()
    mod = args.mod_dir.resolve()
    failures: list[str] = []
    try:
        command_contract = reviewed_recipe_command(recipe)
    except (OSError, ValueError) as exc:
        failures.append(f"reviewed recipe command contract: {exc}")
        command_contract = {"raw_command": "", "wrapper_argv": [], "serve_argv": []}
    mod_failures, mod_manifest_entries = mod_manifest_failures(mod)
    failures.extend(mod_failures)
    failures.extend(artifact_binding_failures(root, FINAL_ARTIFACT_SHA256))
    acceptance_path = root / "acceptance.json"
    runtime_path = root / "runtime.json"
    acceptance = json.loads(acceptance_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    failures.extend(readme_runtime_completion_failures(root, runtime))
    launch_epoch_text = (root / "launch-epoch.txt").read_text()
    launch_epoch = (
        int(launch_epoch_text.strip())
        if re.fullmatch(r"[1-9]\d*\n?", launch_epoch_text)
        else None
    )
    try:
        launch_receipt = json.loads((root / "launch-receipt.json").read_text())
    except (OSError, json.JSONDecodeError):
        launch_receipt = None
    try:
        proxy_registry = json.loads((root / "proxy-models.json").read_text())
    except (OSError, json.JSONDecodeError):
        proxy_registry = None
    failures.extend(
        top_level_metadata_failures(
            acceptance, runtime, launch_epoch, EXPECTED_ACCEPTANCE_RUN_ID
        )
    )
    require(
        proxy_registry == [{
            "api_base": "http://192.168.178.47:8000/v1",
            "max_model_len": 850000,
            "model_name": MODEL,
        }],
        "proxy registry receipt",
        failures,
    )
    proxy_start = (root / "proxy-start.log").read_text()
    proxy_process_commands = []
    for line in proxy_start.splitlines():
        marker = "/home/max/.cache/uv/archive-v0/"
        if marker in line:
            proxy_process_commands.append(line[line.index(marker):])
    require(
        proxy_status_valid(proxy_start)
        and sum(proxy_command_valid(command) for command in proxy_process_commands) == 1,
        "proxy status receipt",
        failures,
    )

    require(acceptance.get("cluster_id") == CLUSTER, "acceptance cluster ID", failures)
    require(
        acceptance.get("invocation_argv") == EXPECTED_ACCEPTANCE_ARGV,
        "acceptance invocation argv",
        failures,
    )
    require(acceptance.get("model") == MODEL, "acceptance model", failures)
    require(acceptance.get("direct_url") == "http://127.0.0.1:8000", "acceptance direct route", failures)
    require(acceptance.get("proxy_url") == "http://127.0.0.1:4000", "acceptance proxy route", failures)
    require(set(acceptance.get("checks", {})) == EXPECTED_CHECKS, "acceptance check set", failures)
    require(acceptance.get("recipe_sha256") == sha(recipe), "recipe SHA", failures)
    require(acceptance.get("acceptance_script_sha256") == sha(root / "acceptance.py"), "acceptance script SHA", failures)
    require(acceptance.get("telemetry_capture_script_sha256") == sha(root / "capture_load_telemetry.py"), "telemetry capture script SHA", failures)
    failures.extend(receipt_interval_failures(acceptance, launch_epoch))

    checks = acceptance.get("checks", {})
    for key in ("direct_models", "proxy_models"):
        row = checks.get(key, {})
        failures.extend(receipt_route_failures(key, row))
        require(row.get("passed") is True, f"{key} verdict", failures)
        require(row.get("http") == 200, f"{key} HTTP status", failures)
        require("error" not in row.get("response", {}), f"{key} response route/schema", failures)

    direct_models = checks.get("direct_models", {}).get("response", {}).get("data", [])
    proxy_models = checks.get("proxy_models", {}).get("response", {}).get("data", [])
    require(any(row.get("id") == MODEL for row in direct_models), "direct model discovery", failures)
    require(any(row.get("id") == MODEL for row in proxy_models), "proxy model discovery", failures)

    verify_chat_receipt("direct_exact", checks.get("direct_exact", {}), exact_request("GLM53_DIRECT_OK"), failures)
    verify_chat_receipt("proxy_exact", checks.get("proxy_exact", {}), exact_request("GLM53_PROXY_OK"), failures)
    require((message(checks.get("direct_exact", {})).get("content") or "").strip() == "GLM53_DIRECT_OK", "direct exact completion", failures)
    require((message(checks.get("proxy_exact", {})).get("content") or "").strip() == "GLM53_PROXY_OK", "proxy exact completion", failures)

    c4 = checks.get("direct_concurrency_c4", {})
    rows = c4.get("results", [])
    markers = [f"GLM53_C4_{index}_OK" for index in range(4)]
    require(c4.get("passed") is True, "direct_concurrency_c4 verdict", failures)
    require(c4.get("markers") == markers and len(rows) == 4, "C4 cardinality", failures)
    if len(rows) == 4:
        for index, (row, marker) in enumerate(zip(rows, markers)):
            verify_chat_receipt(f"direct_concurrency_c4[{index}]", row, exact_request(marker), failures)
        require(all((message(row).get("content") or "").strip() == marker for row, marker in zip(rows, markers)), "C4 exact outputs", failures)
        try:
            require(max(row["started_at"] for row in rows) < min(row["completed_at"] for row in rows), "C4 requests did not overlap", failures)
        except (KeyError, TypeError, ValueError):
            failures.append("C4 timestamp schema")

    long_body, long_prompt = long_request()
    long_row = checks.get("direct_long_context", {})
    verify_chat_receipt("direct_long_context", long_row, long_body, failures)
    require(long_row.get("needle") == "NEEDLE_GL53_842917", "long-context needle metadata", failures)
    require(long_row.get("filler_token") == "alpha " and long_row.get("filler_repeats_each_side") == 55_000, "long-context filler metadata", failures)
    require(long_row.get("prompt_chars") == len(long_prompt) and long_row.get("prompt_sha256") == hashlib.sha256(long_prompt.encode()).hexdigest(), "long-context prompt binding", failures)
    require((message(long_row).get("content") or "").strip() == "NEEDLE_GL53_842917", "long-context retrieval", failures)
    prompt_tokens = long_row.get("response", {}).get("usage", {}).get("prompt_tokens", 0)
    require(isinstance(prompt_tokens, int) and prompt_tokens >= 100_000, "long-context prompt token count", failures)

    load_row = checks.get("direct_telemetry_load", {})
    verify_chat_receipt("direct_telemetry_load", load_row, telemetry_load_request(), failures)
    require(
        load_row.get("response", {}).get("usage", {}).get("completion_tokens") == 512,
        "telemetry load completion tokens",
        failures,
    )

    reasoning_row = checks.get("direct_reasoning_open_stop", {})
    verify_chat_receipt(
        "direct_reasoning_open_stop", reasoning_row, reasoning_open_stop_request(), failures
    )
    reasoning_choice = (reasoning_row.get("response", {}).get("choices") or [{}])[0]
    reasoning_message = reasoning_choice.get("message", {})
    failures.extend(reasoning_stop_failures(reasoning_choice))

    disabled_row = checks.get("direct_thinking_disabled_stop", {})
    verify_chat_receipt(
        "direct_thinking_disabled_stop",
        disabled_row,
        thinking_disabled_stop_request(),
        failures,
    )
    disabled_choice = (disabled_row.get("response", {}).get("choices") or [{}])[0]
    failures.extend(thinking_disabled_stop_failures(disabled_choice))

    tool_row = checks.get("direct_tool_call", {})
    verify_chat_receipt("direct_tool_call", tool_row, tool_request(), failures)
    calls = message(tool_row).get("tool_calls") or []
    exact_calls = []
    for call in calls:
        function = call.get("function", {})
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (json.JSONDecodeError, TypeError):
            arguments = None
        if function.get("name") == "get_weather" and arguments == {"city": "Paris, France"}:
            exact_calls.append(call)
    require(len(calls) == len(exact_calls) == 1, "exact tool arguments", failures)

    expected_fixture = canonical_fixture()
    require(len(expected_fixture) == 716 and hashlib.sha256(expected_fixture).hexdigest() == FIXTURE_SHA256, "internal canonical vision fixture", failures)
    fixture_row = acceptance.get("fixture", {})
    fixture = root / fixture_row.get("path", "__missing_fixture__")
    require(
        fixture_row == {
            "path": "vision-quadrants.png",
            "sha256": FIXTURE_SHA256,
            "size": 716,
            "description": "256x256 RGB PNG: red TL, green TR, blue BL, yellow BR",
        }
        and fixture.is_file()
        and fixture.read_bytes() == expected_fixture,
        "canonical vision fixture bytes",
        failures,
    )
    vision_body = vision_request(expected_fixture)
    for key in ("direct_vision", "proxy_vision"):
        row = checks.get(key, {})
        verify_chat_receipt(key, row, vision_body, failures)
        require((message(row).get("content") or "").strip() == "RED", key.replace("_", " "), failures)
        try:
            url = row["request"]["messages"][0]["content"][1]["image_url"]["url"]
            prefix, payload = url.split(",", 1)
            decoded = base64.b64decode(payload, validate=True)
            require(prefix == "data:image/png;base64" and decoded == expected_fixture, f"{key} exact data URL bytes", failures)
        except Exception:
            failures.append(f"{key} exact data URL bytes")

    expected_ocr_fixture = canonical_ocr_fixture()
    require(
        len(expected_ocr_fixture) == 766
        and hashlib.sha256(expected_ocr_fixture).hexdigest() == OCR_FIXTURE_SHA256,
        "internal canonical OCR fixture",
        failures,
    )
    ocr_fixture_row = acceptance.get("ocr_fixture", {})
    ocr_fixture = root / ocr_fixture_row.get("path", "__missing_ocr_fixture__")
    require(
        ocr_fixture_row == {
            "path": "vision-ocr.png",
            "sha256": OCR_FIXTURE_SHA256,
            "size": 766,
            "description": "544x96 RGB PNG with black bitmap text GLM53 OCR 8429",
        }
        and ocr_fixture.is_file()
        and ocr_fixture.read_bytes() == expected_ocr_fixture,
        "canonical OCR fixture bytes",
        failures,
    )
    ocr_body = ocr_request(expected_ocr_fixture)
    for key in ("direct_ocr", "proxy_ocr"):
        row = checks.get(key, {})
        verify_chat_receipt(key, row, ocr_body, failures)
        require(message(row).get("content") == OCR_TEXT, key.replace("_", " "), failures)
        try:
            url = row["request"]["messages"][0]["content"][1]["image_url"]["url"]
            prefix, payload = url.split(",", 1)
            decoded = base64.b64decode(payload, validate=True)
            require(
                prefix == "data:image/png;base64" and decoded == expected_ocr_fixture,
                f"{key} exact data URL bytes",
                failures,
            )
        except Exception:
            failures.append(f"{key} exact data URL bytes")

    for key in ("direct_remote_media_rejected", "proxy_remote_media_rejected"):
        failures.extend(remote_media_failures(key, checks.get(key, {})))

    expected_video_fixture = canonical_video_fixture()
    require(
        len(expected_video_fixture) == 835
        and hashlib.sha256(expected_video_fixture).hexdigest() == VIDEO_FIXTURE_SHA256,
        "internal canonical video fixture",
        failures,
    )
    video_fixture_row = acceptance.get("video_fixture", {})
    video_fixture = root / video_fixture_row.get("path", "__missing_video_fixture__")
    require(
        video_fixture_row == {
            "path": "video-tiny.gif",
            "sha256": VIDEO_FIXTURE_SHA256,
            "size": 835,
            "mime_type": "image/gif",
            "description": "deterministic two-frame 224x224 inline GIF for video=0 rejection",
        }
        and video_fixture.is_file()
        and video_fixture.read_bytes() == expected_video_fixture,
        "canonical video fixture bytes",
        failures,
    )
    for key in ("direct_video_rejected", "proxy_video_rejected"):
        row = checks.get(key, {})
        failures.extend(video_rejection_failures(key, row, expected_video_fixture))
        try:
            url = row["request"]["messages"][0]["content"][1]["video_url"]["url"]
            prefix, payload = url.split(",", 1)
            decoded = base64.b64decode(payload, validate=True)
            require(
                prefix == "data:image/gif;base64" and decoded == expected_video_fixture,
                f"{key} exact data URL bytes",
                failures,
            )
        except Exception:
            failures.append(f"{key} exact data URL bytes")

    require(runtime.get("cluster_id") == CLUSTER, "runtime cluster ID", failures)
    require(
        launch_epoch is not None and runtime.get("launch_epoch") == launch_epoch,
        "runtime launch epoch binding",
        failures,
    )
    require(
        runtime.get("launch_receipt_sha256") == sha(root / "launch-receipt.json"),
        "runtime immutable launch receipt SHA",
        failures,
    )
    require(
        launch_epoch is not None
        and launch_epoch <= acceptance.get("started_at", -1)
        <= acceptance.get("completed_at", -1)
        <= runtime.get("captured_at", -1)
        <= runtime.get("completed_at", -1),
        "launch/runtime chronology",
        failures,
    )
    require(runtime.get("expected_image_digest") == IMAGE.rsplit("@", 1)[1], "runtime expected image digest", failures)
    require(
        runtime.get("expected_mod_manifest_sha256") == EXPECTED_MOD_MANIFEST_SHA256,
        "runtime expected mod manifest binding",
        failures,
    )
    require(set(runtime.get("hosts", {})) == EXPECTED_HOSTS, "runtime host set", failures)
    require(runtime.get("acceptance_completed_at") == acceptance.get("completed_at"), "runtime/acceptance timestamp join", failures)
    require(runtime.get("capture_script_sha256") == sha(root / "capture_runtime.py"), "runtime capture script SHA", failures)
    require(runtime.get("captured_at", 0) >= acceptance.get("completed_at", 1), "runtime capture order", failures)

    status = runtime.get("status", {})
    require(status.get("returncode") == 0, "SparkRun status command", failures)
    require(status.get("argv") == ["sparkrun", "status", "--cluster", "vacation-pair2", "--json"], "SparkRun status argv", failures)
    try:
        status_json = json.loads(status["stdout"])
        group = status_json["groups"][CLUSTER]
        meta = group["meta"]
        require(status_json.get("total_containers") == 2, "SparkRun container count", failures)
        require(meta["effective_container_image"] == IMAGE, "SparkRun effective image", failures)
        require(set(meta["hosts"]) == EXPECTED_HOSTS, "SparkRun effective hosts", failures)
        expected_status_rows = {
            (host, f"node_{rank}", IMAGE)
            for host, rank in (("192.168.178.47", 0), ("192.168.178.46", 1))
        }
        status_rows = {
            (item.get("host"), item.get("role"), item.get("image"))
            for item in group.get("containers", [])
        }
        require(status_rows == expected_status_rows and all(str(item.get("status", "")).startswith("Up ") for item in group.get("containers", [])), "SparkRun containers running", failures)
        recipe_state = meta.get("recipe_state", {})
        raw_recipe = recipe_state.get("_raw", {})
        require(recipe_state.get("_applied_overrides") == {}, "SparkRun recipe overrides", failures)
        require(raw_recipe.get("defaults") == IMMUTABLE_DEFAULTS, "SparkRun immutable recipe defaults", failures)
        raw_command = raw_recipe.get("command")
        failures.extend(
            command_binding_failures(command_contract, raw_command, launch_receipt)
        )
        require(
            raw_command == command_contract.get("raw_command"),
            "SparkRun exact reviewed raw command",
            failures,
        )
        require(meta.get("tensor_parallel") == 2 and meta.get("pipeline_parallel") == 1 and meta.get("port") == 8000, "SparkRun topology/port", failures)
        require(meta.get("executor_config") == {
            "cap_add": ["IPC_LOCK"],
            "entrypoint": "",
            "shm_size": "32gb",
            "ulimit": ["memlock=-1:-1", "stack=67108864:67108864", "nofile=65535:65535"],
            "user": "root",
        }, "SparkRun executor security", failures)
    except Exception:
        failures.append("SparkRun status payload")

    for key, url in (("direct_models", "http://127.0.0.1:8000/v1/models"), ("proxy_models", "http://127.0.0.1:4000/v1/models")):
        row = runtime.get(key, {})
        require(
            row.get("requested_url") == url
            and row.get("effective_url") == url
            and row.get("path") == "/v1/models",
            f"runtime {key} route",
            failures,
        )
        require(row.get("http") == 200, f"runtime {key} HTTP status", failures)
        try:
            body = json.loads(row.get("body", ""))
            require(any(item.get("id") == MODEL for item in body.get("data", [])) and "error" not in body, f"runtime {key} model/schema", failures)
        except (json.JSONDecodeError, TypeError):
            failures.append(f"runtime {key} model/schema")

    proxy_status = runtime.get("proxy_status", {})
    require(
        proxy_status.get("argv") == ["sparkrun", "proxy", "status"]
        and proxy_status.get("returncode") == 0
        and proxy_status_valid(proxy_status.get("stdout", "")),
        "runtime proxy status",
        failures,
    )
    proxy_listener = runtime.get("proxy_listener", {})
    require(
        proxy_listener.get("probe_argv") == ["python3", "-c", LISTENER_PROBE, "4000"],
        "proxy listener probe provenance",
        failures,
    )
    failures.extend(proxy_binding_failures(runtime))

    fatal_kernel = (
        "NV_ERR_NO_MEMORY", "NVRM: Xid",
        "Out of memory: Killed process", "oom-kill:",
    )
    required_symbols = (
        "exl3_moe",
        "exl3_fat_gemm",
        "exl3_fat_gemm_scatter",
        "exl3_fat_moe_gateup",
        "exl3_fat_moe_down",
        "exl3_fat_moe_gather",
    )
    expected_host_ranks = {"192.168.178.47": 0, "192.168.178.46": 1}
    runtime_env_by_host: dict[str, str] = {}
    roce_records_by_host: dict[str, object] = {}
    docker_launch_facts: dict[str, dict[str, object]] = {}
    for host, row in runtime.get("hosts", {}).items():
        rank = expected_host_ranks[host]
        container = f"{CLUSTER}_node_{rank}"
        require(row.get("ssh_host") == host and row.get("rank") == rank, f"{host} host/rank metadata", failures)
        require(row.get("container") == container, f"{host} expected container", failures)
        require(row.get("earlyoom", {}).get("stdout", "").strip() == "inactive", f"{host} earlyoom", failures)
        inspect_cmd = row.get("docker_inspect", {})
        require(inspect_cmd.get("returncode") == 0, f"{host} docker inspect", failures)
        container_id: str | None = None
        try:
            inspect = json.loads(inspect_cmd["stdout"])[0]
            container_id = inspect.get("Id")
            docker_launch_facts[host] = {
                "Created": inspect.get("Created"),
                "StartedAt": inspect.get("State", {}).get("StartedAt"),
            }
            require(inspect.get("Name") == f"/{container}", f"{host} inspect container", failures)
            require(
                isinstance(container_id, str)
                and re.fullmatch(r"[0-9a-f]{64}", container_id) is not None,
                f"{host} inspect container ID",
                failures,
            )
            require(inspect.get("Image") == IMAGE_CONFIG_DIGEST, f"{host} image config digest", failures)
            require(inspect["Config"]["Image"] == IMAGE, f"{host} image pin", failures)
            require(inspect["Config"].get("User") == "root", f"{host} container user", failures)
            require(inspect["State"].get("Running") is True and inspect["State"].get("Status") == "running", f"{host} container running", failures)
            host_config = inspect["HostConfig"]
            require(host_config.get("Privileged") is False, f"{host} privileged mode", failures)
            require(host_config.get("NetworkMode") == "host", f"{host} network mode", failures)
            require(host_config.get("IpcMode") == "host", f"{host} IPC mode", failures)
            require(set(host_config.get("SecurityOpt") or []) == {"no-new-privileges", "label=disable"}, f"{host} security options", failures)
            require(host_config.get("CapAdd") == ["CAP_IPC_LOCK"], f"{host} capabilities", failures)
            require(host_config.get("Devices") == [{"PathOnHost": "/dev/infiniband", "PathInContainer": "/dev/infiniband", "CgroupPermissions": "rwm"}], f"{host} RDMA device", failures)
            binds = host_config.get("Binds") or []
            require(
                len(binds) == len(EXPECTED_BINDS) and set(binds) == EXPECTED_BINDS,
                f"{host} exact binds",
                failures,
            )
            mounts = inspect.get("Mounts") or []
            mount_rows = {
                (
                    mount.get("Type"),
                    mount.get("Source"),
                    mount.get("Destination"),
                    mount.get("RW"),
                    mount.get("Mode"),
                    mount.get("Propagation"),
                )
                for mount in mounts
            }
            require(
                len(mounts) == len(EXPECTED_MOUNTS) and mount_rows == EXPECTED_MOUNTS,
                f"{host} exact mounts",
                failures,
            )
        except Exception:
            failures.append(f"{host} inspect payload")
        image_inspect_row = row.get("image_inspect", {})
        try:
            image_inspect = json.loads(image_inspect_row.get("stdout", ""))
            require(len(image_inspect) == 1, f"{host} image inspect payload", failures)
            image_identity = image_inspect[0]
            require(
                image_identity.get("Id") == IMAGE_CONFIG_DIGEST,
                f"{host} image inspect config digest",
                failures,
            )
            require(
                image_identity.get("RepoDigests") == [IMAGE],
                f"{host} image inspect RepoDigests",
                failures,
            )
        except Exception:
            failures.append(f"{host} image inspect payload")
        remote_commands = (
            "container_discovery", "earlyoom", "docker_inspect", "image_inspect", "docker_top", "docker_logs",
            "serve_log", "serve_markers", "runtime_env", "roce_records", "runtime_overlay", "runtime_mod_manifest",
            "serving_process", "pid_namespace", "wrapper_lineage", "nccl_loaded",
            "rdma", "memory", "kernel_since_launch",
            "kernel_readiness_to_acceptance", "kernel_after_acceptance",
        )
        for command in remote_commands:
            command_row = row.get(command, {})
            require(command_row.get("returncode") == 0, f"{host} {command} command", failures)
        discovery = row.get("container_discovery", {})
        require(discovery.get("stdout", "").split() == [container] and CLUSTER in " ".join(discovery.get("argv", [])), f"{host} container discovery binding", failures)
        expected_command = expected_runtime_command(rank, command_contract)
        serving = row.get("serving_process", {}).get("stdout", "")
        serving_match = re.fullmatch(r"pid=(\d+)\ncommand=(.+)\n?", serving)
        serving_container_pid = int(serving_match.group(1)) if serving_match else None
        require(
            serving_match is not None and serving_match.group(2) == expected_command,
            f"{host} serving PID/command binding",
            failures,
        )
        nccl_row = row.get("nccl_loaded", {})
        nccl = nccl_row.get("stdout", "")
        nccl_values = dict(
            re.findall(r"^(pid|path|sha256|package_version)=(.+)$", nccl, re.MULTILINE)
        )
        nccl_pid = (
            int(nccl_values["pid"])
            if re.fullmatch(r"[1-9]\d*", nccl_values.get("pid", ""))
            else None
        )
        docker_top_text = row.get("docker_top", {}).get("stdout")
        serving_host_pid = docker_top_runtime_host_pid(
            docker_top_text, rank, command_contract
        )
        wrapper_host_pid = docker_top_wrapper_host_pid(
            docker_top_text, rank, command_contract
        )
        wrapper_parent_host_pid = docker_top_wrapper_parent_host_pid(
            docker_top_text, rank, command_contract
        )
        if serving_host_pid is None or serving_container_pid is None:
            failures.append(f"{host} serving PID namespace prerequisites")
        else:
            failures.extend(
                f"{host} {reason}"
                for reason in pid_namespace_failures(
                    row.get("pid_namespace"),
                    host,
                    serving_host_pid,
                    serving_container_pid,
                )
            )
        if (
            wrapper_host_pid is None
            or wrapper_parent_host_pid is None
            or container_id is None
        ):
            failures.append(f"{host} wrapper lineage prerequisites")
        else:
            failures.extend(
                f"{host} {reason}"
                for reason in wrapper_lineage_failures(
                    row.get("wrapper_lineage"),
                    host,
                    wrapper_host_pid,
                    wrapper_parent_host_pid,
                    container_id,
                )
            )
        listener_container_pid: int | None = None
        direct_listener_row = row.get("direct_listener")
        if rank == 0:
            try:
                direct_listener = json.loads(
                    direct_listener_row.get("stdout", "")
                    if isinstance(direct_listener_row, dict)
                    else ""
                )
            except (json.JSONDecodeError, TypeError):
                direct_listener = {}
            expected_listener_metadata = {
                "namespace": "container",
                "host": host,
                "container": container,
                "rank": 0,
                "endpoint": {"port": 8000},
            }
            listeners = direct_listener.get("listeners") if isinstance(direct_listener, dict) else None
            listener = listeners[0] if isinstance(listeners, list) and len(listeners) == 1 else {}
            require(
                isinstance(direct_listener_row, dict)
                and direct_listener_row.get("returncode") == 0
                and direct_listener_row.get("stderr") == ""
                and direct_listener_row.get("argv")
                == expected_remote_argv(
                    host, direct_listener_command(container, host, rank)
                ),
                "head direct listener command",
                failures,
            )
            require(
                {key: direct_listener.get(key) for key in expected_listener_metadata}
                == expected_listener_metadata
                and set(direct_listener) == {*expected_listener_metadata, "listeners"}
                and listener.get("bind") == "0.0.0.0"
                and bool(re.fullmatch(r"[1-9]\d*", str(listener.get("inode", ""))))
                and listener.get("command") == expected_command
                and isinstance(listener.get("pid"), int),
                "head direct listener endpoint",
                failures,
            )
            listener_container_pid = (
                listener.get("pid") if isinstance(listener.get("pid"), int) else None
            )
            require(
                serving_container_pid is not None
                and listener_container_pid == serving_container_pid,
                "head direct listener serving PID",
                failures,
            )
        else:
            failures.extend(worker_listener_failures(direct_listener_row, host, container))
        for reason in docker_top_inventory_failures(
            docker_top_text,
            rank,
            serving_host_pid,
            serving_container_pid,
            listener_container_pid,
            nccl_pid,
            command_contract,
        ):
            failures.append(f"{host} {reason}")
        if serving_container_pid is not None and launch_epoch is not None:
            expected_commands = expected_host_commands(
                host,
                container,
                str(serving_container_pid),
                launch_epoch,
                runtime.get("acceptance_completed_at", -1),
                runtime.get("readiness_epoch", -1),
            )
            for key, expected_argv in expected_commands.items():
                require(
                    row.get(key, {}).get("argv") == expected_argv,
                    f"{host} {key} exact command",
                    failures,
                )
        else:
            failures.append(f"{host} exact command prerequisites")
        for kernel_name in (
            "kernel_since_launch",
            "kernel_readiness_to_acceptance",
            "kernel_after_acceptance",
        ):
            require(
                row.get(kernel_name, {}).get("stderr") == "",
                f"{host} {kernel_name} stderr",
                failures,
            )
        overlay_row = row.get("runtime_overlay", {})
        require(
            overlay_row.get("returncode") == 0
            and overlay_row.get("argv") == expected_runtime_overlay_argv(host, container),
            f"{host} runtime_overlay exact command",
            failures,
        )
        overlay = overlay_row.get("stdout", "")
        overlay_stderr = overlay_row.get("stderr", "")
        require(
            "[OK] complete GLM-5.3 runtime patched state verified" in overlay,
            f"{host} runtime patch live gate",
            failures,
        )
        require("FATAL" not in overlay_stderr, f"{host} runtime overlay FATAL stderr", failures)
        require("request-state-bound reasoning stop policy OK" in overlay, f"{host} stop-guard live gate", failures)
        require(all(symbol in overlay for symbol in required_symbols), f"{host} E3 symbols", failures)
        manifest_row = row.get("runtime_mod_manifest", {})
        expected_manifest_stdout = "\n".join(
            [f"manifest_sha256={EXPECTED_MOD_MANIFEST_SHA256}"]
            + [f"{entry}: OK" for entry in mod_manifest_entries]
        ) + "\n"
        require(
            manifest_row.get("returncode") == 0
            and manifest_row.get("stdout") == expected_manifest_stdout
            and manifest_row.get("stderr") == "",
            f"{host} complete runtime mod manifest verification",
            failures,
        )
        failures.extend(
            f"{host} {reason}"
            for reason in serve_marker_failures(
                row.get("serve_markers"), host, container
            )
        )
        env = row.get("runtime_env", {}).get("stdout", "")
        runtime_env_by_host[host] = env
        for expected in ("ABLIT=0", "GLM53_ADAPTIVE_K=off", "GLM53_DENSE_FP8=off", "GLM53_INDEXER_WORKSPACE=rightsize", "EXL3_FAT_GROUPED=1"):
            require(expected in env, f"{host} env {expected}", failures)
        try:
            roce_records_by_host[host] = json.loads(
                row.get("roce_records", {}).get("stdout", "")
            )
        except (json.JSONDecodeError, TypeError):
            roce_records_by_host[host] = None
        nccl_row = row.get("nccl_loaded", {})
        nccl = nccl_row.get("stdout", "")
        nccl_values = dict(re.findall(r"^(pid|path|sha256|package_version)=(.+)$", nccl, re.MULTILINE))
        require(
            serving_container_pid is not None
            and nccl_values.get("pid") == str(serving_container_pid),
            f"{host} NCCL serving PID",
            failures,
        )
        require(nccl_values.get("path") == NCCL_PATH, f"{host} canonical NCCL path", failures)
        require(nccl_values.get("sha256") == NCCL_SHA256, f"{host} canonical NCCL digest", failures)
        require(nccl_values.get("package_version") == "2.30.7", f"{host} NCCL package", failures)
        nccl_command = " ".join(nccl_row.get("argv", []))
        require(
            container in nccl_command
            and serving_container_pid is not None
            and f"/proc/{serving_container_pid}/maps" in nccl_command,
            f"{host} NCCL SSH/container/PID binding",
            failures,
        )
        rdma = row.get("rdma", {}).get("stdout", "")
        require(parse_rdma(rdma) == EXPECTED_RDMA, f"{host} exact HCA map", failures)
        after = row.get("kernel_after_acceptance", {}).get("stdout", "")
        require(not any(pattern in after for pattern in fatal_kernel), f"{host} post-acceptance kernel safety", failures)
        bounded = row.get("kernel_readiness_to_acceptance", {}).get("stdout", "")
        require(
            not any(pattern in bounded for pattern in fatal_kernel),
            f"{host} readiness-to-acceptance kernel safety",
            failures,
        )

    failures.extend(
        launch_epoch_failures(
            launch_epoch,
            launch_receipt,
            docker_launch_facts,
            sha(recipe),
            command_contract,
        )
    )

    head = runtime.get("hosts", {}).get("192.168.178.47", {})
    head_container = f"{CLUSTER}_node_0"
    for key, filename in (
        ("postready_rc", "glm53-postready.rc"),
        ("postready_ok", "glm53-postready.ok"),
        ("postready_log", "glm53-postready.log"),
    ):
        receipt = head.get(key, {})
        require(
            receipt.get("returncode") == 0
            and receipt.get("argv")
            == expected_postready_argv("192.168.178.47", head_container, filename),
            f"post-ready {key} exact command",
            failures,
        )
    require(head.get("postready_rc", {}).get("stdout", "").strip() == "0", "post-ready rc", failures)
    require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\S+", head.get("postready_ok", {}).get("stdout", "").strip())), "post-ready receipt", failures)
    require("postready gate OK" in head.get("postready_log", {}).get("stdout", ""), "post-ready success log", failures)
    try:
        receipt_readiness_epoch = int(timestamp(head["postready_ok"]["stdout"].strip()))
    except (KeyError, TypeError, ValueError):
        receipt_readiness_epoch = -1
    require(
        runtime.get("readiness_epoch") == receipt_readiness_epoch
        and isinstance(launch_epoch, int)
        and launch_epoch <= receipt_readiness_epoch <= acceptance.get("started_at", -1),
        "readiness epoch binding",
        failures,
    )

    try:
        telemetry = parse_telemetry(root / "load-telemetry.log")
    except (OSError, ValueError):
        telemetry = {}
        failures.append("telemetry grammar")
    require(set(telemetry) == {"before", "during", "after"}, "telemetry sample set", failures)
    acceptance_start = acceptance.get("started_at", float("inf"))
    acceptance_end = acceptance.get("completed_at", float("-inf"))
    timestamps = [telemetry.get(name, {}).get("timestamp") for name in ("before", "during", "after")]
    require(
        all(isinstance(value, (int, float)) for value in timestamps)
        and timestamps[0] <= acceptance_start <= timestamps[1] <= acceptance_end <= timestamps[2]
        and timestamps[0] < timestamps[1] < timestamps[2],
        "telemetry acceptance window",
        failures,
    )
    load_started = load_row.get("started_at")
    load_completed = load_row.get("completed_at")
    require(
        isinstance(load_started, (int, float))
        and isinstance(load_completed, (int, float))
        and load_started <= load_completed,
        "telemetry direct-load interval",
        failures,
    )
    # telemetry direct-load overlap is enforced independently for every host acquisition.
    for host in EXPECTED_HOSTS:
        during_host = telemetry.get("during", {}).get("hosts", {}).get(host, {})
        acquisition_started = during_host.get("acquisition_started_at")
        acquisition_completed = during_host.get("acquisition_completed_at")
        require(
            all(
                isinstance(value, (int, float))
                for value in (
                    load_started,
                    load_completed,
                    acquisition_started,
                    acquisition_completed,
                )
            )
            and acquisition_started <= acquisition_completed
            and acquisition_started <= load_completed
            and acquisition_completed >= load_started,
            f"during host acquisition overlap {host}",
            failures,
        )
    for sample_name in ("before", "during", "after"):
        sample = telemetry.get(sample_name, {})
        require(set(sample.get("hosts", {})) == EXPECTED_HOSTS, f"{sample_name} telemetry hosts", failures)
        if sample_name == "during":
            require(sample.get("running", 0) >= 1, "during active request", failures)
        for host, host_row in sample.get("hosts", {}).items():
            acquisition_started = host_row.get("acquisition_started_at")
            acquisition_completed = host_row.get("acquisition_completed_at")
            require(
                isinstance(acquisition_started, (int, float))
                and isinstance(acquisition_completed, (int, float))
                and acquisition_started <= acquisition_completed,
                f"{sample_name} {host} acquisition interval",
                failures,
            )
            gpu_samples = host_row.get("gpu_samples", [])
            if sample_name == "during":
                require(
                    len(gpu_samples) == 10
                    and all(
                        gpu.get("pstate") == "P0"
                        and gpu.get("utilization", 0) >= 90
                        for gpu in gpu_samples
                        if isinstance(gpu, dict)
                    ),
                    f"during {host} GPU load",
                    failures,
                )
            hcas = host_row.get("hcas", {})
            for reason in roce_binding_failures(
                host,
                runtime_env_by_host.get(host, ""),
                roce_records_by_host.get(host),
                set(hcas),
            ):
                require(False, f"{host} {reason}", failures)
            for hca in ("rocep1s0f1", "roceP2p1s0f1"):
                require("200 Gb/sec" in hcas.get(hca, {}).get("rate", "") and "ACTIVE" in hcas.get(hca, {}).get("state", ""), f"{sample_name} {host} {hca} link", failures)
    if set(telemetry) == {"before", "during", "after"}:
        for host in EXPECTED_HOSTS:
            for hca in ("rocep1s0f1", "roceP2p1s0f1"):
                before = telemetry["before"].get("hosts", {}).get(host, {}).get("hcas", {}).get(hca, {})
                after = telemetry["after"].get("hosts", {}).get(host, {}).get("hcas", {}).get(hca, {})
                require(isinstance(before.get("xmit"), int) and isinstance(before.get("rcv"), int) and isinstance(after.get("xmit"), int) and isinstance(after.get("rcv"), int) and after["xmit"] > before["xmit"] and after["rcv"] > before["rcv"], f"{host} {hca} traffic delta", failures)

    manifest = root / "SHA256SUMS"
    if manifest.is_file():
        declared: dict[str, str] = {}
        try:
            for line in manifest.read_text().splitlines():
                digest, rel = line.split(None, 1)
                rel = rel.lstrip("* ").removeprefix("./")
                require(rel not in declared, f"manifest duplicate {rel}", failures)
                declared[rel] = digest
        except ValueError:
            failures.append("manifest syntax")
        require(set(declared) == EXPECTED_MANIFEST, "manifest exact file set", failures)
        for rel, digest in declared.items():
            path = root / rel
            require(path.is_file() and sha(path) == digest, f"manifest {rel}", failures)
    else:
        failures.append("SHA256SUMS missing")

    if failures:
        print(json.dumps({"passed": False, "failures": failures}, indent=2))
        return 1
    print(json.dumps({"passed": True, "checks": len(EXPECTED_CHECKS), "hosts": sorted(EXPECTED_HOSTS), "prompt_tokens": prompt_tokens}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
