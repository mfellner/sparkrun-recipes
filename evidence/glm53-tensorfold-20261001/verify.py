#!/usr/bin/env python3
"""Recompute semantic TensorFold acceptance from raw HTTP receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "GLM-5.3-Flash-EXL3"
CLUSTER = "sparkrun_98fa271a0dea8179_9e378c82c913"
IMAGE_ID = "sha256:0266e5767a4fb7cba90e587c6f69ab59dea24a0cd08ea983d333295717e9bf24"
IMAGE_DIGEST = "sha256:22789f0cb3dc308f0b2ce52a33961b88bd624af1725e91e8aba0a74a671bb969"


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def reply(row: dict) -> dict:
    require(row["http"] == 200 and row["response"].get("model") == MODEL, "HTTP/model mismatch")
    require(row["url"] in ("http://192.168.178.47:8000/v1/chat/completions", "http://192.168.178.47:4000/v1/chat/completions"), "route")
    require("error" not in row["response"] and len(row["response"]["choices"]) == 1, "schema/error")
    require(row["finished_at"] > row["started_at"] > 0, "request timing")
    message = row["response"]["choices"][0]["message"]
    require(row["response"]["choices"][0]["finish_reason"] in ("stop", "tool_calls"), "finish reason")
    require(row["response"]["usage"]["completion_tokens"] > 0, "empty generation")
    return message


def verify(data: dict) -> dict:
    require(data["cluster_id"] == CLUSTER and data["completed_at"] > data["captured_at"], "capture identity")
    tests = data["tests"]
    all_replies = [tests["direct"], tests["proxy"], tests["vision"], tests["tool"], tests["long_needle"], *tests["concurrent"]]
    all_http = [*all_replies, data["models"]["direct"], data["models"]["proxy"],
                data["health_before"], data["health_after"], data["long_tokenize"]]
    require(all(data["captured_at"] <= row["started_at"] < row["finished_at"] <= data["completed_at"]
                for row in all_http), "HTTP event outside workload acceptance interval")
    require(all(row["url"] == "http://192.168.178.47:8000/v1/chat/completions"
                for row in [tests["direct"], tests["vision"], tests["tool"], tests["long_needle"], *tests["concurrent"]]), "direct route")
    require(tests["proxy"]["url"] == "http://192.168.178.47:4000/v1/chat/completions", "proxy route")
    require(data["health_before"]["finished_at"] <= min(
                [row["started_at"] for row in all_replies] + [data["long_tokenize"]["started_at"]]),
            "health_before not before acceptance requests")
    require(max(data["models"][kind]["finished_at"] for kind in ("direct", "proxy")) <=
            data["health_before"]["started_at"], "model discovery not before health_before")
    require(data["health_after"]["started_at"] >= max(row["finished_at"] for row in all_replies), "health_after not after requests")
    require(data["long_tokenize"]["finished_at"] <= tests["long_needle"]["started_at"], "tokenize must precede long completion")
    require(all(row["request"]["model"] == MODEL for row in all_replies), "requested model")
    for field, path in (
        ("recipe_sha256", ROOT / "recipes/glm-5.3-flash-exl3-tensorfold-dual-spark-1m.yaml"),
        ("mod_manifest_sha256", ROOT / "mods/glm53-tensorfold-1m/SHA256SUMS"),
    ):
        require(data[field] == hashlib.sha256(path.read_bytes()).hexdigest(), f"{field} drift")
    require(set(data["hosts"]) == {"192.168.178.47", "192.168.178.46"}, "rank hosts")
    ids = set()
    for rank, host in enumerate(("192.168.178.47", "192.168.178.46")):
        row = data["hosts"][host]
        require(row["rank"] == rank and row["container"] == CLUSTER + f"_node_{rank}", "rank/container")
        require(row["id"] not in ids and len(row["id"]) >= 64, "unique container IDs")
        ids.add(row["id"])
        require(row["image_id"] == IMAGE_ID and any(x.endswith(IMAGE_DIGEST) for x in row["repo_digests"]), "image pin")
        require(row["privileged"] is False and row["network_mode"] == "host" and row["ipc_mode"] == "host", "container security")
        require(row["user"] in ("", "root"), "user")
    for kind, port in (("direct", 8000), ("proxy", 4000)):
        row = data["models"][kind]
        require(row["url"] == f"http://192.168.178.47:{port}/v1/models", kind + " models route")
        require(row["http"] == 200 and row["body"].get("object") == "list" and
                "error" not in row["body"] and [x["id"] for x in row["body"]["data"]] == [MODEL], kind + " models")
    for kind in ("health_before", "health_after"):
        row = data[kind]
        require(row["url"] == "http://192.168.178.47:8000/health" and row["http"] == 200, kind + " HTTP/route")
    before, after = data["health_before"]["body"], data["health_after"]["body"]
    require(before["ok"] is True and after["ok"] is True and
            before["backend"] == after["backend"] == "tensorfold", "health")
    require(before["context_length"] == after["context_length"] == 1048576, "context")
    require(before["streams"]["max"] == after["streams"]["max"] == 4, "parallel")
    require(after["pool_tokens"] >= 1048576 and after["requests_running"] == 0, "pool/idle")
    tests = data["tests"]
    for key, marker, port in (("direct", "DIRECT_OK", 8000), ("proxy", "PROXY_OK", 4000)):
        row = tests[key]
        require(row["url"].startswith(f"http://192.168.178.47:{port}/"), key + " route")
        require(row["request"]["messages"][0]["content"] == f"Reply with exactly {marker}.", key + " request")
        require((reply(row).get("content") or "").strip() == marker, key + " semantics")
    wave = tests["concurrent"]
    require(len(wave) == 4 and max(r["started_at"] for r in wave) < min(r["finished_at"] for r in wave), "not C4 overlapping")
    for i, row in enumerate(wave):
        marker = f"WAVE_{i}"
        require(row["request"]["messages"][0]["content"] == f"Reply with exactly {marker}.", "C4 request")
        require((reply(row).get("content") or "").strip() == marker, "C4 response")
    vision = tests["vision"]
    parts = vision["request"]["messages"][0]["content"]
    require(vision["request"]["messages"][0]["role"] == "user" and len(parts) == 2 and
            parts[0]["type"] == "image_url" and parts[1] == {"type": "text", "text":
            "What color fills the LEFT HALF of the image? Answer one color word."}, "vision question")
    image_url = parts[0]["image_url"]["url"]
    require(image_url.startswith("data:image/png;base64,"), "image input")
    import base64
    fixture = base64.b64decode(image_url.split(",", 1)[1], validate=True)
    require(hashlib.sha256(fixture).hexdigest() == "fc75966897d50143d883d0f9cc09b6b508989a4a6dcb03650983dba508377741" == data["fixture_png_sha256"], "image fixture")
    require((reply(vision).get("content") or "").strip().lower().strip(" .!\n") == "red", "vision semantics")
    tool = tests["tool"]
    request = tool["request"]
    require(request["messages"] == [{"role": "user", "content":
            "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."}], "tool question")
    require(len(request.get("tools", [])) == 1 and request["tools"][0]["type"] == "function", "offered tool")
    offered = request["tools"][0]["function"]
    require(offered["name"] == "add_tags" and offered["parameters"] == {
        "type": "object", "required": ["doc_id", "tags"],
        "properties": {"doc_id": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}}}, "tool schema")
    calls = reply(tool).get("tool_calls") or []
    require(len(calls) == 1 and calls[0]["function"]["name"] == "add_tags", "tool name")
    args = json.loads(calls[0]["function"]["arguments"])
    require(args == {"doc_id": 42, "tags": ["urgent", "finance", "q3"]}, "typed tool arguments")
    long_row = tests["long_needle"]
    needle = "SABLE-ORCHID-7294"
    archive = "\n".join(f"Archive row {i}: blue quartz transit ledger 1847 stable."
                        + (f" The only verification code is {needle}." if i == 1837 else "")
                        for i in range(3300))
    expected_prompt = "Find the only verification code in this archive. Reply with the exact code only.\n" + archive
    require(long_row["request"]["messages"] == [{"role": "user", "content": expected_prompt}], "complete long fixture")
    require(data["long_tokenize"]["url"] == "http://192.168.178.47:8000/tokenize" and
            data["long_tokenize"]["request"] == {"model": MODEL, "messages": long_row["request"]["messages"]}, "tokenize request join")
    tokens = data["long_tokenize"]["body"]
    require(data["long_tokenize"]["http"] == 200 and tokens["count"] == 50152 and
            len(tokens["tokens"]) == tokens["count"] and tokens["max_model_len"] == 1048576, "long prompt tokenized")
    require(long_row["response"]["usage"]["prompt_tokens"] == 50146, "long API prompt usage")
    require((reply(long_row).get("content") or "").strip().strip(" .!") == "SABLE-ORCHID-7294", "needle retrieval")
    return {"passed": True, "hosts": len(data["hosts"]), "concurrency": len(wave),
            "long_prompt_tokens": data["long_tokenize"]["body"]["count"], "pool_tokens": after["pool_tokens"]}


if __name__ == "__main__":
    print(json.dumps(verify(json.loads((HERE / "receipt.json").read_text())), indent=2))
