#!/usr/bin/env python3
import argparse
import base64
import json
import urllib.request


def post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--image", default="/tmp/glm52-vision-test.png")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    results = {}

    with urllib.request.urlopen(base + "/v1/models", timeout=30) as response:
        models = json.load(response)
    ids = [item.get("id") for item in models.get("data", [])]
    assert "glm-5.2" in ids, ids
    results["models"] = ids

    semantic = post(base + "/v1/chat/completions", {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "Reply with exactly GLM52_SEMANTIC_OK and nothing else."}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    semantic_text = semantic["choices"][0]["message"]["content"].strip()
    assert semantic_text == "GLM52_SEMANTIC_OK", repr(semantic_text)
    results["semantic"] = {"content": semantic_text, "usage": semantic.get("usage")}

    tool = post(base + "/v1/chat/completions", {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "Use the get_weather tool for Berlin. Do not answer directly."}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    message = tool["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    assert calls and calls[0]["function"]["name"] == "get_weather", message
    tool_args = json.loads(calls[0]["function"]["arguments"])
    assert tool_args.get("city", "").lower() == "berlin", tool_args
    results["tool_call"] = {"name": calls[0]["function"]["name"], "arguments": tool_args, "usage": tool.get("usage")}

    with open(args.image, "rb") as handle:
        image_data = base64.b64encode(handle.read()).decode()
    vision = post(base + "/v1/chat/completions", {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_data}},
            {"type": "text", "text": "Describe the large background shape/color and the centered shape/color in one short sentence."},
        ]}],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    vision_text = vision["choices"][0]["message"]["content"].strip()
    lower = vision_text.lower()
    assert all(word in lower for word in ("red", "square", "white", "circle")), vision_text
    results["vision"] = {"content": vision_text, "usage": vision.get("usage")}

    records = []
    for index in range(6000):
        marker = ""
        if index == 37:
            marker = " ALPHA_MARKER_7319"
        elif index == 5963:
            marker = " OMEGA_MARKER_2846"
        records.append(f"record={index:05d} inert payload cedar quartz cobalt{marker}")
    long_prompt = "\n".join(records) + "\nReturn exactly: ALPHA_MARKER_7319|OMEGA_MARKER_2846"
    long_result = post(base + "/v1/chat/completions", {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": long_prompt}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    long_text = long_result["choices"][0]["message"]["content"].strip()
    assert long_text == "ALPHA_MARKER_7319|OMEGA_MARKER_2846", repr(long_text)
    results["long_context"] = {"content": long_text, "usage": long_result.get("usage")}

    anthropic = post(base + "/v1/messages", {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "What is 2 plus 2? Reply with just the numeral."}],
        "max_tokens": 32,
        "temperature": 0,
    })
    blocks = anthropic.get("content") or []
    anthropic_text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text").strip()
    assert anthropic_text == "4", repr(anthropic_text)
    results["anthropic"] = {"content": anthropic_text, "usage": anthropic.get("usage")}

    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
