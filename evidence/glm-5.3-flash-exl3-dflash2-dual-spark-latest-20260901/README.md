# GLM-5.3 EXL3 latest-refresh evidence — 2026-09-01

This directory is the compact publication receipt for the refresh of
`recipes/glm-5.3-flash-exl3-dflash2-dual-spark-1m.yaml` to:

- Upstream source: `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks@c190db1ae17ba8dff20129ed1f308d10c63cf37d`
- Main model: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw@024db9f7e9871e8efdf21538ba55af7442be3cd5`
- DFlash2 draft: `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a`
- Image: `ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks@sha256:4f30fba4248ed7d78dddda37d9ffc0fea5cc8c567c5dd98042ad808efdde9791`
- SparkRun: `0.3.6`
- Final process: `sparkrun_9000d029f34ee753_fe02508ba745`
- Hosts: `dgx01` / `192.168.178.47` and `dgx02` / `192.168.178.46`

## What changed

The refresh pins upstream's compiled E2 fat-expert kernel, `MAX_NUM_BATCHED_TOKENS=7168`, unconditional Reasoning-Effort prefix, guarded indexer workspace, numeric spinwait control, changed TP2 DFlash2 payload, latest EXL3 overlay, K-pool tail clamp, and optional ABLIT hook with `ABLIT=0`. The exact arm64 image was built from c190db1, published anonymously from GHCR, pinned by manifest digest, and exercised live through E2 direct/scatter on both ranks. SparkRun uses `GPU_MEM_UTIL=0.86` plus upstream's opt-in `GLM53_INDEXER_WORKSPACE=rightsize`; live logs show 4,909.5 MiB reclaimed per rank and enough KV for the configured 1M context.

`raw/hf-pins.json` contains complete deterministic path/size/blob-or-LFS-SHA maps for both old/new main and draft revisions. The recomputed maps prove that the main mirror's serving payload did not change from the previous pin while the draft `model.safetensors` did. It also joins all 120 mirror LFS shard hashes and ten core serving files to `brandonmusic@5ab363a8` materialization receipts/manifest hashes, proving the logical serving payload is byte-identical. `generate_hf_pin_maps.py` regenerates every map and comparison from the immutable revisions. `rollback-recipe-32db610.yaml` preserves the preceding recipe.

## Empirical acceptance on the exact final process

`verification-summary.json` is generated fail-closed by `verify.py`, which verifies complete artifact checksums and recomputes request hashes, timing joins, raw response semantics, model-file comparisons, APC counters, and per-rank runtime identities. `test_verify_negative_controls.py` also proves that corrupted raw responses/counters/NCCL evidence are rejected even if producer `passed` fields and the altered artifact's manifest digest are preserved. It proves:

- Direct and LiteLLM-proxied model discovery and exact semantic completions
- Four concurrent direct semantic completions
- A 110,042-prompt-token retrieval with exact needle recovery
- Tool calling and deterministic still-image understanding
- Direct and proxied deterministic video understanding
- C4 thinking-enabled JSON-schema termination plus a non-thinking control
- APC reuse with a positive prefix-cache-hit increase recomputed from timestamped raw before/after Prometheus counter lines
- Reasoning continued past a client `Question:` stop string and produced the required final answer
- A forced 2,300-completion-token request completed after the K-pool tail fix
- Rank-0 post-readiness gate receipt `rc=0`
- Both containers remained running, each host exposed two active 200 Gb/s RDMA links, and the loaded NCCL library on each rank independently matched package 2.30.7 and the pinned library digest
- Per-rank E2 diagnostics proved `configured_tier=kernel`, `effective_tier=kernel`, TP ranks 0/1, compiled symbols present, and positive direct/scatter execution counters
- Captured serve/kernel logs contained no selected fatal, Xid, or OOM-kill signatures. Three bounded head-side `NV_ERR_NO_MEMORY` notices occurred during post-ready acceptance pressure; none occurred after final acceptance, and both ranks remained healthy

The configured context is 1,000,000 tokens; the largest semantic request rerun for this refresh was 110,042 prompt tokens. Historical 950K and throughput measurements belong to preceding pinned processes and are not claimed for this refresh.

## Files

- `acceptance.py`, `acceptance.json`: direct/proxy, C4, 110K, tool and image matrix
- `acceptance_xgrammar.py`, `acceptance-xgrammar.json`: structured-output reasoning C4 and control
- `acceptance_latest_features.py`, `acceptance-latest-features.json`: APC, forced stop, video and 2,300-token K-pool generation
- `capture_image_public.py`, `raw/image-public.json`: retained anonymous GHCR manifest bytes plus local/worker/package command receipts
- `capture_runtime_evidence.py`, `raw/runtime-success.json`: exact containers, commands, logs, kernel window, RDMA state, loaded NCCL identities and gate receipts
- `generate_hf_pin_maps.py`, `raw/hf-pins.json`: reproducible per-file immutable-revision maps and comparisons
- `verify.py`, `test_verify_negative_controls.py`, `verification-summary.json`: semantics-recomputing assertions, adversarial controls and compact result
- `raw/dry-run-output.json`, `raw/dry-run.txt`, `raw/launch-output.json`, `raw/launch.log`, `raw/upstream-head.txt`, `raw/image-tag-inspect.txt`, `raw/image-public.json`, `raw/hf-pins.json`: provenance and launch receipts. The JSON files preserve exact command stdout; the text copies normalize only trailing blank/space presentation. SparkRun's placement token is environment-generated and is not expected to equal the live workload suffix.
- `SHA256SUMS`: hashes of every evidence file except itself
