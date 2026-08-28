# GLM-5.3 Flash EXL3 DFlash2 TP2 acceptance — 2026-08-28

This directory contains the publication evidence for
`recipes/glm-5.3-flash-exl3-dflash2-dual-spark-900k.yaml`.

## Immutable scope

- Recipe SHA-256: `add8d048335618b0d0f9be221bb7038e7e968c2c6f5ce0606e0c4d3332070aff`
- Upstream launcher source: `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks@4676496e8d4622aaeb0675d79eb15ee1f26c1950`
- Main model: `brandonmusic/GLM-5.3-Flash-tr3-4bpw@5ab363a8dcf6405955fd5f99671e01a1c9fb124b`
- Draft: `incoai/GLM-5.3-Flash-DFlash2@7d74cdd881ed7e32c31175984a67823127b66cfe`
- Image: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58`
- Hosts: `dgx01` / `192.168.1.95` and `dgx02` / `192.168.1.111`
- Successful SparkRun ID: `sparkrun_73f38a238771a1ea_f80511f7c8ce`

The upstream repository contains rounded README summaries but no raw sparkDash result receipts. The comparison below therefore means **parity with upstream-reported values under the documented protocol**, not independent validation of upstream raw measurements.

After this acceptance run, upstream `main` advanced to `f3043c95bbf95fb91dd160fe58d740cd152a02c3` with additional scheduler/reasoning/JIT-cache source patches. At publication time, GHCR `:exl3` still resolved to the tested digest `9bb1557…`; those later source patches were not present in the published image and were not mixed into this evidence. This artifact remains deliberately scoped to exact source `4676496…` plus the exact tested image digest.

## Performance protocol

`benchmark_parity.py` preserves every warm-up and measured row. Requests use:

- Structured count 1→200, code (`clamp_00`…`clamp_49`), or hash-map prose prompts
- `temperature=0`, `top_p=1`, thinking off
- Streaming with authoritative response usage
- 400 forced output tokens, `ignore_eos=true`
- Per-stream decode: `(completion_tokens - 1) / (last_visible - first_visible)`
- Aggregate decode: total decode tokens over the earliest-first to latest-last visible-token window
- One warm-up per concurrency shape and five measured waves

### Structured parity

| Concurrency | Upstream stream | Measured stream median (range) | Upstream aggregate | Measured aggregate median (range) | Ratio | Result |
|---:|---:|---:|---:|---:|---:|---|
| C1 | 62.9 | 65.763 (65.154–66.878) | 62.9 | 65.763 (65.154–66.879) | 1.046× | Pass |
| C2 | 51.7 | 49.868 (45.207–50.689) | 103.3 | 99.352 (89.625–100.416) | 0.962× aggregate | Pass |
| C4 | 37.1 | 39.447 (36.082–40.078) | 146.5 | 155.363 (137.356–157.940) | 1.060× aggregate | Pass |

The fail-closed parity floor was fixed before measurement at 95% for both mean-stream and aggregate medians. All three shapes pass.

### Workload controls

- Code C1: **62.531 tok/s** median (61.824–62.821)
- Code C4 aggregate: **145.692 tok/s** median (140.021–149.949)
- Prose C1: **28.093 tok/s** median (26.595–31.100), versus upstream's reported 26.9

These controls demonstrate the expected content-dependent DFlash2 acceptance behavior rather than treating one structured output as universal model TPS.

## Functional acceptance

`acceptance.json` records all raw API responses. Every check passed:

- Direct and proxied `/v1/models`
- Direct and proxied exact semantic completion
- Four concurrent direct semantic completions
- GLM 4.7 tool call
- Deterministic direct and proxied vision fixture
- Needle retrieval at **110,035 prompt tokens**

The server is configured for 900,000 tokens. The largest empirically completed semantic request in this evidence is 110,035 prompt tokens; no near-900K request was attempted.

## Runtime and telemetry

- vLLM `0.1.dev20051+g487ecf187`, CUDA 13.0, NCCL 2.30.7
- TP2 over both active 200 Gb/s CX-7 HCAs per host
- Fused `exl3_moe`, DFlash2 k=7, `fp8_ds_mla`, captured target and speculator graphs
- Live KV pool: **1,056,593 tokens**; configured context: 900,000
- Containers were non-privileged, root, host-networked/IPC, `no-new-privileges`, 32 GiB SHM
- Telemetry window: 470 samples per host; 419 busy samples per host
- Busy GPU utilization median: 96% on both hosts
- SM clock median: 2,405 MHz on both hosts
- Power median: 42.28 W on dgx01 and 45.38 W on dgx02
- Both HCA rails carried traffic; zero recorded HCA receive errors or transmit discards

CUDA-graph probing emitted bounded `NV_ERR_NO_MEMORY` notices from 18:25:01 through 18:25:16. They stopped before readiness. Subsequent parity, proxy, vision, tool, concurrency, and 110K semantic tests passed. Scans found no Xid, OOM kill, or fatal runtime-log match after readiness. See `runtime-health.json` for the scoped result.

## Artifacts

- `benchmark_parity.py` — exact benchmark harness
- `benchmark-structured.json` — structured warm-ups and measured waves
- `benchmark-code.json` — code warm-ups and measured waves
- `benchmark-prose.json` — prose warm-up and measured waves
- `collect_telemetry.py` / `telemetry-parity.jsonl` — timestamped all-rank GPU/HCA samples
- `acceptance.py` / `acceptance.json` — semantic, concurrency, tool, vision, long-context and proxy requests plus raw responses
- `vision-quadrants.png` — deterministic multimodal fixture
- `capture_runtime_evidence.py` / `raw/runtime-success.json` — exact successful-run commands, per-rank inspect/env/logs, bounded kernel window, link state/rate and post-capture API markers
- `run-manifest.json` — hash-bound join from the successful SparkRun ID to benchmark, acceptance, telemetry and runtime artifacts
- `runtime-health.json` — concise effective runtime/security/environment summary derived from the raw capture

Run the committed harnesses to reproduce the requests, and use `SHA256SUMS` plus `run-manifest.json` to verify the exact raw artifact set before independently recomputing published statistics.
