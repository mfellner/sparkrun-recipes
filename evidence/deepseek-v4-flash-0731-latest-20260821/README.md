# DeepSeek V4 Flash 0731 functional acceptance evidence

## Scope

This directory preserves the functional acceptance artifacts used for the refresh of:

- launcher source: `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
- launcher revision: `a462a9e541c684b58c7f380bbd92c7d851557f31`
- model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- served alias: `deepseek-v4-flash-0731`
- direct API used by the harnesses: `http://127.0.0.1:8000/v1`

The model was served by the two-node `dgx01`/`dgx02` SparkRun deployment. This package deliberately makes only the functional claims mechanically supported by the committed harnesses and result files below.

## Exact-response, tool, context, and concurrency acceptance

### Tool battery

Command:

```bash
python harnesses/tool-battery.py \
  http://127.0.0.1:8000/v1/chat/completions \
  deepseek-v4-flash-0731
```

Artifacts:

- harness: `harnesses/tool-battery.py`
- output: `tool-battery.txt`

Result: **7/7 passed**—single call, nested schema, multi-turn replay, parallel calls, thinking plus tool use, issue-55 truncation behavior, and forced tool choice. The output records valid JSON tool arguments and `finish_reason=length` with no malformed tool call in the truncation case.

### Semantic ladder and concurrent soak

Command:

```bash
python harnesses/stability-quick.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model deepseek-v4-flash-0731 \
  --skip-vl \
  --ladder 8192,32768,65536,131072 \
  --decode-tokens 64 \
  --soak-minutes 2 \
  --soak-prompt-tokens 32768 \
  --soak-concurrency 4 \
  --output stability-quick.json
```

Artifacts:

- harness: `harnesses/stability-quick.py`
- result: `stability-quick.json`

The machine-readable report has `ok: true` and an empty `failures` array. It records:

- readiness smoke response `OK`;
- semantic sentinels at 8,216, 32,792, 65,561, and 131,096 prompt tokens;
- two successful C4 soak rounds at 32,789 prompt tokens per request;
- all eight concurrent responses returning their expected `SOAK_OK` sentinel.

Vision was intentionally skipped because this recipe is the upstream default text-only profile.

### RULER-lite retrieval

Command:

```bash
python harnesses/ruler-lite.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model deepseek-v4-flash-0731 \
  --lengths 262144 \
  --tasks sniah \
  --thinking-key thinking \
  --max-tokens 256 \
  --request-timeout 1800 \
  --output ruler-lite-262k.json
```

Artifacts:

- harness: `harnesses/ruler-lite.py`
- result: `ruler-lite-262k.json`

Result: **PASS** at 262,168 actual prompt tokens. The gold and predicted value are both `24592`.

The configured context is 1,048,576 tokens. The largest semantic request established by this evidence package is 262,168 prompt tokens; near-1M semantic validation is not claimed.

## sparkDash visibility

Artifacts:

- structured snapshot: `sparkdash-dgx01-metrics.json`
- rendered browser capture: `sparkdash-dgx01-detail.png`

The structured snapshot reports `dgx01` online and the LLM as available with backend `vllm`, model ID `deepseek-v4-flash-0731`, and context length `1048576`. The rendered capture visibly shows the `dgx01` head and `dgx02` worker tabs, model `deepseek-v4-flash-0731`, port 8000, context `1,048,576`, and engine `Active`.

## Acceptance boundary

This compact package does **not** claim or establish:

- a universal decode-performance number or parity result;
- a near-1M semantic request;
- vision/VL-sidecar behavior;
- API-key modes;
- LiteLLM proxy routing;
- reboot or Docker-daemon recovery;
- independently recomputable negative kernel/runtime-log claims.

Operational checks outside this committed package were used to keep the live deployment safe, but they are not promoted into publication claims here.

`SHA256SUMS` covers every evidence artifact other than the checksum file itself.
