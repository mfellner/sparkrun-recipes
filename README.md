# mfellner SparkRun Recipes

Custom [SparkRun](https://sparkrun.dev/) recipe registry for NVIDIA DGX Spark and compatible GB10 systems.

## Add the registry

```bash
sparkrun registry add --trust https://github.com/mfellner/sparkrun-recipes.git
```

Recipes are then available under the `@mfellner/` namespace. Review the repository before granting trust: several recipes intentionally require custom images, host networking/IPC, device access, capabilities, or lifecycle hooks. If the registry was already added without trust, run `sparkrun registry trust mfellner` after auditing it, or add `--trust` explicitly to a local-file launch.

## Recipes

### GLM 5.3 Flash EXL3 + DFlash2 — dual Spark, 1M context (recommended)

`@mfellner/glm-5.3-flash-exl3-dflash2-dual-spark-1m`

Latest immutable SparkRun adaptation of [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), audited against upstream commit `c190db1ae17ba8dff20129ed1f308d10c63cf37d`.

- Exactly two DGX Spark or compatible GB10 nodes using native `vllm-distributed` TP2/MP
- `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` pinned to `024db9f7e9871e8efdf21538ba55af7442be3cd5`; deterministic maps prove its serving payload is unchanged from the prior 25a44fdb mirror pin and byte-identical to the original `brandonmusic` 5ab logical serving payload
- Updated DFlash2 draft payload pinned independently to `bf582e4eacc1810f76656d1811693ff6c6737d2a`
- Public arm64 CUDA 13 image built from the exact upstream commit and pinned at `ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks@sha256:4f30fba4248ed7d78dddda37d9ffc0fea5cc8c567c5dd98042ad808efdde9791`
- Checksum-pinned mod validates every selected RoCE GID and compiled E2 symbol, then applies each fail-closed overlay on every rank before serve; the complete source/static suite runs separately against the exact staged bundle
- A fail-closed local follow-up ignores client-provided stop strings for this default-thinking profile so EOS/`max_tokens` govern completion; an environment opt-out restores stock stop behavior
- Persistent Triton, TileLang, TorchInductor, FlashInfer, and vLLM caches plus a post-readiness DFlash2/sampler shape sweep
- FP8 `fp8_ds_mla` target KV, TP2-sharded DFlash2 k=7, fused EXL3 MoE, CUDA graphs, prefix caching, image/video, GLM 4.7 tools, and GLM 4.5 reasoning
- `MAX_MODEL_LEN=1000000`, `MAX_NUM_SEQS=4`, upstream E2 `MAX_NUM_BATCHED_TOKENS=7168`, and SparkRun `GPU_MEM_UTIL=0.86`; the one-point memory adjustment keeps vLLM's admission gate fail-closed after per-rank overlay checks while retaining a rendered KV estimate above 1M tokens

Port `8000` is the intentional SparkRun/sparkDash/LiteLLM adaptation from upstream `8888`; the served alias remains `GLM-5.3-Flash-EXL3`.

The published profile runs as root with host networking/IPC, `no-new-privileges`, `CAP_IPC_LOCK`, and `/dev/infiniband`; Docker privileged mode is off. Its unauthenticated multimodal API and distributed-runtime ports must remain on a trusted firewalled private network.

#### Latest refresh acceptance (2026-09-01)

The exact refreshed recipe launched under SparkRun 0.3.6 as `sparkrun_9000d029f34ee753_fe02508ba745`. The rank-0 post-readiness gate completed with `rc=0`, and final live receipts confirmed the c190db1 image/source pin, main/draft snapshots, TP2 DFlash2, 7,168-token prefill, right-sized guarded indexer workspace, unconditional Reasoning-Effort template, E2 kernel tier with positive direct/scatter counters on both ranks, ABLIT disabled, loaded NCCL 2.30.7, and two active 200 Gb/s RDMA links per host.

Direct and LiteLLM-proxied discovery/completions, C4 exact concurrency, C4 thinking-enabled JSON Schema plus a non-thinking control, tool calling, still-image and video understanding, APC reuse, the forced reasoning-stop case, and a **110,042-prompt-token** retrieval all passed. A forced **2,300-completion-token** request exercised the corrected K-pool tail path. Three bounded head-side allocation notices occurred during post-ready acceptance pressure, but none occurred after final acceptance; both ranks remained running with no selected fatal, Xid, or host OOM-kill signatures. The committed [latest-refresh evidence](evidence/glm-5.3-flash-exl3-dflash2-dual-spark-latest-20260901/) contains raw requests/responses, per-rank E2/NCCL/process joins, deterministic model maps, public-image provenance, checksums, a semantics-recomputing verifier, and adversarial negative controls.

The configured context is 1,000,000 tokens; 110,042 prompt tokens is the largest semantic request rerun for this refresh. Historical 950K and throughput results below were not rerun on the c190db1 process and are not claimed for it.

#### Previous 1M acceptance (2026-08-29 profile)

The measurements below predate the c190db1 E2 kernel, 7,168-token prefill, right-sized indexer workspace, latest DFlash2 payload, and template/spinwait refresh. They remain historical evidence for the prior pinned process, not performance evidence for the current recipe.

The exact final recipe launched under SparkRun 0.3.6 as `sparkrun_d56463b6b176dd58_009b8bd44277`. Logs proved padded slot-share, hybrid APC with only the drafter SWA group marked EAGLE, fused EXL3 MoE, FP8 sparse-MLA KV, updated DFlash weights, captured target/speculator graphs, and NCCL 2.30.7 over both CX-7 rails. The final live KV pool was **1,814,136 tokens**, providing **1.81×** maximum concurrency for a 1,000,000-token request.

Five measured structured waves after a preserved warm-up at each shape were collected on the precursor process with the same image, weights, serve command, 1M memory profile and core performance patches; final changes add launch/preflight/video/stop-guard/XGrammar correctness rather than model-execution tuning:

| Concurrency | Stream median | Aggregate median | Upstream-reported aggregate | Ratio |
|---:|---:|---:|---:|---:|
| C1 | 63.428 tok/s | 63.428 tok/s | 62.9 tok/s | 1.008× |
| C2 | 50.836 tok/s | 101.668 tok/s | 103.3 tok/s | 0.984× |
| C4 | 41.098 tok/s | 161.262 tok/s | 146.5 tok/s | 1.101× |

Code C4 reached **167.539 aggregate tok/s**. Prose C1 measured **25.683 tok/s**, 95.5% of upstream's reported 26.9.

All direct/proxy semantic, C4 concurrency, tool, still-image, and video checks passed. The latest behavioral fixes were validated end to end:

- Single follow-up APC reuse: **7,168 tokens**
- Four concurrent follow-ups: **28,672 total cached tokens**
- Reasoning continued past a client stop string and returned the required final answer
- **3×256,029-prompt-token** requests all completed successfully
- Near-limit needle retrieval passed at **950,037 prompt tokens** in 1,084 seconds

The configured context is 1,000,000 tokens; 950,037 prompt tokens is the largest empirically completed semantic request. The exact final process reran direct/proxy completion, C4, tool, image, video, APC, the previously failing reasoning-stop request, and a 110,035-token retrieval; the expensive 3×256K and 950K tests were not repeated after the launch-overlay corrections. The committed [1M evidence](evidence/glm-5.3-flash-exl3-dflash2-dual-spark-1m-20260829/) preserves both process IDs/checksums, every measured row, requests/responses, APC/occupancy/near-limit receipts, and correlated precursor telemetry.

Run it on the saved pair:

```bash
sparkrun run @mfellner/glm-5.3-flash-exl3-dflash2-dual-spark-1m \
  --cluster YOUR_CLUSTER \
  --no-follow
```

This endpoint is unauthenticated, root-user, host-networked/IPC, multimodal, and trust-gated. Use only on a firewalled private network; do not expose inference, native-distributed, or NCCL ports publicly. Prefer inline media and authenticated/filtering ingress for untrusted clients. Keep `earlyoom` inactive while loaded and restore it only after unloading and verifying safe RAM/swap headroom.

### GLM 5.3 Flash EXL3 + DFlash2 — dual Spark, 900K context (legacy rollback)

`@mfellner/glm-5.3-flash-exl3-dflash2-dual-spark-900k`

An immutable SparkRun adaptation of [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), audited against upstream commit `4676496e8d4622aaeb0675d79eb15ee1f26c1950`.

- Exactly two DGX Spark or compatible GB10 nodes using SparkRun's native `vllm-distributed` runtime (`mp`, TP2, one rank per node)
- `brandonmusic/GLM-5.3-Flash-tr3-4bpw` pinned to revision `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`
- DFlash2 draft `incoai/GLM-5.3-Flash-DFlash2` independently pinned to revision `7d74cdd881ed7e32c31175984a67823127b66cfe`
- MiaAI-Lab's arm64 CUDA 13 EXL3 overlay image pinned at manifest digest `sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58`
- EXL3/TR3 4-bpw routed experts, fused `exllamav3_ext.exl3_moe`, and the upstream NoPE sparse-MLA SM121 patches
- FP8 `fp8_ds_mla` KV cache, 900,000-token configured context, GPU utilization `0.87`, four sequences, and 1,024-token prefill chunks
- DFlash2 k=7 with probabilistic draft sampling, standard rejection sampling, BF16/automatic draft KV, and draft TP1
- CUDA graphs with upstream capture sizes `1 2 4 8 16 24 32`, prefix caching, and FlashInfer autotuning disabled
- Image input was empirically validated; video limits/template and the upstream video-placeholder installer are configured but the video request path was not empirically tested
- Fail-closed GPU overlay self-checks on every rank before vLLM starts; video patch installation runs at launch but its deferred runtime hook is not advertised as fail-closed
- Persistent TorchInductor, vLLM, FlashInfer, and temporary compilation caches under SparkRun's per-host cache mount

Port `8000` is an intentional operational adaptation from upstream's `8888`, preserving this registry's SparkRun discovery, sparkDash, and LiteLLM conventions. It does not change the model execution profile or benchmark accounting. The served model alias remains upstream-compatible: `GLM-5.3-Flash-EXL3`.

#### Live acceptance

The recipe was launched on the `dgx01`/`dgx02` ASUS GX10 pair with SparkRun ID `sparkrun_73f38a238771a1ea_f80511f7c8ce`. The live stack used vLLM `0.1.dev20051+g487ecf187`, CUDA 13.0, NCCL 2.30.7, both 200 Gb/s CX-7 rails, fused EXL3 MoE, DFlash2 k=7, FP8 sparse-MLA KV, and captured target/speculator graphs. The measured KV pool was **1,056,593 tokens** for the configured 900,000-token context.

Five measured structured waves after a preserved warm-up at each shape delivered:

| Concurrency | Stream median | Aggregate median | Upstream-reported aggregate | Ratio |
|---:|---:|---:|---:|---:|
| C1 | 65.763 tok/s | 65.763 tok/s | 62.9 tok/s | 1.046× |
| C2 | 49.868 tok/s | 99.352 tok/s | 103.3 tok/s | 0.962× |
| C4 | 39.447 tok/s | 155.363 tok/s | 146.5 tok/s | 1.060× |

All three concurrency shapes passed the predeclared 95% parity floor for both mean-stream and aggregate medians. Code C4 reached 145.692 aggregate tok/s; the deliberately less predictable prose workload measured 28.093 tok/s versus upstream's reported 26.9.

Direct and LiteLLM-proxied exact completion, C4 semantic concurrency, GLM tool calling, deterministic image understanding, and a **110,035-prompt-token** needle retrieval all passed. The configured context is 900,000 tokens; 110,035 prompt tokens is the largest empirically completed semantic request, and near-900K validation was not performed. The committed [acceptance evidence](evidence/glm-5.3-flash-exl3-dflash2-dual-spark-900k-20260828/) preserves every warm-up and measured row, submitted acceptance requests, raw API responses, correlated all-rank telemetry, successful-runtime captures, recipe checksums, and the workload-to-artifact manifest.

Run it on a saved two-node cluster:

```bash
sparkrun run @mfellner/glm-5.3-flash-exl3-dflash2-dual-spark-900k \
  --cluster YOUR_CLUSTER \
  --no-follow
```

Or specify the two hosts directly, with the API head first:

```bash
sparkrun run @mfellner/glm-5.3-flash-exl3-dflash2-dual-spark-900k \
  --hosts HEAD_IP,WORKER_IP \
  --no-follow
```

#### Operational requirements

- This profile needs most of each GB10's unified memory. Stop unrelated GPU workloads first and verify free memory on both ranks. Keep `earlyoom` inactive while the loaded model remains inside its kill threshold; restore it only after unloading and confirming safe RAM/swap headroom.
- Validate management reachability, MTU 9000, and bidirectional jumbo traffic on every intended CX-7 path before launch. SparkRun supplies per-host `NODE_IP`/`VLLM_HOST_IP`, socket interfaces, HCA selection, and RoCE GID values instead of copying upstream's kit-specific interface names.
- SparkRun auto-selects an available native-distributed rendezvous port, preferring `25000`, instead of upstream's fixed `29521`. This changes only rank coordination; it does not change the model execution profile.
- The first launch downloads approximately 164 GiB of target weights plus a 2.3 GiB draft and distributes the pinned 9.8 GB compressed image. Weight loading, CUDA-graph capture, and lazy kernel compilation can take substantial time; active compiler processes and growing caches are progress, not a readiness failure.
- If GitHub Container Registry blob delivery is unusually slow, pre-pull the pinned base on the head before launching: `docker pull vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`. The final pinned GHCR image reuses 30 shared base layers (9.68 GB compressed) and then needs only its approximately 108 MB of non-base layers; SparkRun can ship the completed final image to the worker over CX-7.
- The DFlash2 checkpoint is licensed CC BY-NC-ND 4.0 for research/evaluation. The EXL3 checkpoint uses ShapleyMCG License 1.0. Review both before deployment.

#### Security

The endpoint binds to `0.0.0.0:8000` without API authentication and supports image/video requests, including URL media accepted by vLLM. The validated SparkRun 0.3.6 rootless Docker path is non-privileged with `no-new-privileges`, but this recipe explicitly runs as container user `root` and uses host networking, host IPC, GPU CDI, `/dev/infiniband`, and `IPC_LOCK`. **Run it only on a trusted, firewalled private network; never expose inference, native-distributed rendezvous, or NCCL ports directly to the public Internet.** Prefer inline `data:` media and place authenticated ingress plus an outbound media allowlist/filter in front of vLLM for untrusted clients.

### DeepSeek V4 Flash 0731 DSpark — dual Spark, 1M context (recommended)

`@mfellner/deepseek-v4-flash-0731-dspark-dual-spark-1m`

An immutable SparkRun adaptation of [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark). The effective text-serving stack and vendored hotfix bundle were audited against upstream commit `a462a9e541c684b58c7f380bbd92c7d851557f31`.

- Two DGX Spark or compatible GB10 nodes using SparkRun's native `vllm-distributed` runtime
- `deepseek-ai/DeepSeek-V4-Flash-0731` at revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Anemll DSpark image `0.1.1`, pinned at manifest digest `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
- Tensor parallelism 2, 1,048,576-token configured context, and NVFP4 DS-MLA KV cache
- Native DSpark speculative decoding with five probabilistically sampled draft tokens
- FlashInfer B12X MXFP4 MoE, SM121 kernels, async scheduling, chunked prefill, prefix caching, and FlashInfer autotuning
- DeepSeek V4 reasoning and tool-call parsers, with reasoning effort `max` by default; clients can disable thinking per request
- Exact upstream default hotfixes for long-context NVFP4 dispatch, MTP memory, scheduler fairness/concurrent prefills, prefix-cache retention, structured-output reasoning boundaries, tool truncation, stop handling, TileLang JIT resilience, and lower GB10 IPC spin load
- A checksum-pinned adjacent SparkRun mod copies and applies those hotfixes checksum-verified and fail-closed on both ranks before vLLM starts
- Encoder compatibility patch copied from the pinned model snapshot only after SHA-256 verification
- Upstream's current text profile: GPU utilization `0.835`, 1,024-token long-prefill chunks, two in-flight partial prefills, persistent TileLang/runtime caches, and a 1,800-second model-execution timeout for lazy JIT
- 64 GiB shared memory, RoCE v2/NCCL environment, and readiness plus semantic-completion post-launch gates

SparkRun starts vLLM through a detached container exec, so Docker restart policy cannot reconstruct serving after a daemon or host reboot. Relaunch this workload through SparkRun after reboot rather than treating a restored `sleep infinity` container as healthy.

Port `8000` is an intentional operational adaptation from upstream's `8888`, preserving SparkRun registry discovery, sparkDash, and existing proxy conventions. It does not change model execution or benchmark methodology. The served model name remains upstream-compatible: `deepseek-v4-flash-0731`.

The latest upstream repository also contains optional alternatives that this recipe does not enable by default: an abliterated checkpoint, experimental VL sidecar/MCP path, assistant-final continuation patch, and GPU `thinking_token_budget` extension. API-key modes are not implemented by this recipe; use trusted-network controls or add and independently audit secret-backed authentication before exposure. The published recipe remains the official text-only checkpoint and matches upstream's default-off gates for the included optional patches.

Run it on a saved two-node cluster:

```bash
sparkrun run @mfellner/deepseek-v4-flash-0731-dspark-dual-spark-1m \
  --cluster YOUR_CLUSTER \
  --no-follow
```

Or specify the two hosts directly:

```bash
sparkrun run @mfellner/deepseek-v4-flash-0731-dspark-dual-spark-1m \
  --hosts HEAD_IP,WORKER_IP \
  --no-follow
```

#### Live acceptance

The refreshed recipe passed direct exact-response inference, all 7 tool-calling cases, semantic context checks at 8K/32K/64K/131K, two C4 concurrent 32K soak rounds, and a RULER-lite single-needle retrieval at **262,168 prompt tokens**. The configured context is 1,048,576 tokens; 262,168 prompt tokens is the largest empirically completed semantic request in this refresh, and near-1M validation was not performed.

The committed [latest-upstream acceptance evidence](evidence/deepseek-v4-flash-0731-latest-20260821/) contains the exact functional harnesses and machine-readable results, plus structured and rendered sparkDash snapshots. Performance output retained in that directory is exploratory session data, not a published parity claim.

#### Operational requirements

- Temporarily stop `earlyoom` on both nodes before launch. This memory-saturated workload can otherwise lose a rank during model load or CUDA-graph capture. Keep it inactive while the model remains loaded; restore it after unloading and confirming safe memory/swap headroom.
- The head must be listed first and serves the API on port `8000`. SparkRun supplies per-rank `VLLM_HOST_IP` plus auto-detected HCA, socket interface, and RoCEv2 GID values; validate every intended fabric path before launch.
- First startup or a new request shape can spend minutes in CUDA/TileLang/FlashInfer compilation. The recipe persists these caches and extends the model-execution timeout to 1,800 seconds; active compilation is forward progress, not a readiness failure.
- Stock SparkRun 0.3.1 gives the initial API-port gate only 240 seconds, while upstream allows 1,500 seconds. Cold startup of this profile exceeded the stock gate but continued and became healthy. Use a SparkRun release with a configurable/1,500-second readiness budget or apply the equivalent local `wait_for_port` budget correction before relying on the launcher exit code; always inspect the live process and logs before stopping an actively loading model.

#### Security and operational notes

The endpoint binds to `0.0.0.0:8000` without API authentication and executes pinned model remote code via `--trust-remote-code`. The validated SparkRun 0.3.1 rootless Docker path is non-privileged with `no-new-privileges`, but uses host networking, host IPC, GPU CDI, and `/dev/infiniband`. **Run it only on a trusted, firewalled private network; never expose inference or NCCL/bootstrap ports directly to the public Internet.**

During first-time/lazy CUDA graph and kernel-shape warmup, the GB10 driver can emit recoverable `NV_ERR_NO_MEMORY` allocation notices even when vLLM completes graph capture and remains healthy. Treat an API failure, Xid, worker exit, or recurring post-warmup allocation failure as a fault; do not treat the kernel notice alone as a successful or failed stress test.

### DeepSeek V4 Flash base checkpoint — dual Spark, 1M context (legacy)

`@mfellner/deepseek-v4-flash-dual-spark-1m`

A SparkRun adaptation of [MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context](https://github.com/MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context). It uses:

Upstream synchronization: `ffa38f9e1bea5c06a6195fca9bff17c04f4785da` (latest `main` fetched 2026-08-04).

- Two DGX Spark or compatible GB10 nodes
- SparkRun's proven `vllm-ray` multi-node orchestration
- Tensor parallelism across both GPUs
- 1,000,000-token configured context
- FP8 KV cache
- MTP speculative decoding
- Prefix caching and FlashInfer autotuning
- DeepSeek V4 reasoning and tool-call parsers
- The SparkRun-compatible local `vllm-node` image, while retaining MiaAI-Lab's model, FP8 KV-cache, 1M-context, MTP, and FlashInfer tuning
- Hugging Face checkpoint and remote-code revision `60d8d70770c6776ff598c94bb586a859a38244f1`, used in distribution and the served snapshot path

Inspect and estimate memory:

```bash
sparkrun show @mfellner/deepseek-v4-flash-dual-spark-1m
sparkrun recipe vram @mfellner/deepseek-v4-flash-dual-spark-1m
```

Run on a saved two-node cluster:

```bash
sparkrun run @mfellner/deepseek-v4-flash-dual-spark-1m \
  --cluster YOUR_CLUSTER \
  --no-follow
```

Or specify the two hosts directly:

```bash
sparkrun run @mfellner/deepseek-v4-flash-dual-spark-1m \
  --hosts HEAD_IP,WORKER_IP \
  --no-follow
```

#### Security

The DeepSeek endpoint binds to `0.0.0.0:8000` without API authentication and executes the pinned model revision's remote code via `--trust-remote-code`. With the validated SparkRun 0.3.1 rootless Docker path, its containers run non-privileged as the host user with `no-new-privileges`, while still using host networking and host IPC on both nodes. **Run it only on a trusted, firewalled private network; do not expose inference, Ray, or NCCL ports to the public Internet.** Review a new Hugging Face revision before changing the recipe's `model_revision`, distribution revision, and served snapshot path together, and re-check effective container security settings after a SparkRun upgrade.

### Unsloth Qwen3.6 35B A3B NVFP4 — single Spark, 256K context

`@mfellner/unsloth-qwen3.6-35b-a3b-nvfp4-dgx-spark`

A SparkRun adaptation of [MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark](https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark). It uses:

Upstream synchronization: `79c9e6f359f6101cdacd0dfd6fe9861ae2493a4d` (latest `main` fetched 2026-08-04).

- One DGX Spark or compatible GB10 node
- `unsloth/Qwen3.6-35B-A3B-NVFP4`
- MiaAI-Lab's image with OCI version label `v0.26.0-gb10.2` (not a published image tag), pinned at manifest digest `sha256:19627342e1da2607f4db50745dca30e57d7dd0ebff06062f03fd69b43a252931`
- SM121-native FlashInfer B12X linear kernels with MiaAI-Lab's soft fallback to automatic kernel selection for unsupported FP8 and other non-NVFP4 layers
- Hugging Face checkpoint and remote-code revision `739af1e7aac320af1682ed1e0cce369af4c5265d`, used in distribution and the served snapshot path
- 262,144-token configured context and FP8 KV cache
- MTP speculative decoding with two draft tokens
- Async scheduling and chunked prefill
- Qwen3 reasoning, Qwen3-coder tool calling, and vision input
- No prefix caching, following upstream's warning about the experimental Mamba-layer path

Run it on a single host:

```bash
sparkrun run @mfellner/unsloth-qwen3.6-35b-a3b-nvfp4-dgx-spark \
  --hosts HOST_IP \
  --no-follow
```

The recipe explicitly clears the entrypoint so SparkRun controls the container lifecycle, while retaining the upstream image, model, patched mixed-quant kernel selection, and vLLM flags. It persists FlashInfer/vLLM compilation caches through SparkRun's host cache mount, avoiding a full kernel rebuild after normal recipe restarts. The pinned current image has no baked entrypoint or Docker healthcheck.

#### Security

This upstream-compatible recipe binds vLLM to `0.0.0.0`, does not configure API authentication, permits URL-based vision input from any media domain, and executes the pinned model revision's remote code via `--trust-remote-code`. The recipe explicitly requests user `root` inside the container, but the validated SparkRun 0.3.1 rootless Docker path remains non-privileged with `no-new-privileges`; it still uses host networking and host IPC. **Run it only on a trusted, firewalled private network. Do not expose port 8000 to the public Internet.** In untrusted or multi-tenant environments, restrict ingress with a host firewall or reverse proxy, add authentication, and replace `--allowed-media-domains '*'` with an explicit allowlist before deployment. Review a new Hugging Face revision before changing the recipe's `model_revision`, distribution revision, and served snapshot path together, and re-check effective container security settings after a SparkRun upgrade.

### GLM-5.2 Vision NVFP4+AQLM — Path A, maximum context

`@mfellner/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-348k`

A SparkRun adaptation of [MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks). It uses:

Upstream synchronization for both GLM profiles: `5c85163ccb8d98d395880e71e2dbd03976a3f4ad` (latest `main` fetched 2026-08-06). The upstream launcher fail-closes if `HF_REVISION` falls back to mutable Hugging Face `main`. Hugging Face head `c5d93567f1ff2de4dbba6018b58a653654c1309a` was also audited: everything after `53e0082eedebd806b63e19779c47905937d768ca` is README and Terminal-Bench trace material, with no serving weight, index, config, tokenizer, template, or remote-code change. These SparkRun recipes therefore pin `53e0082…` as the latest serving checkpoint in `model_revision`, distribution, and the served snapshot path, avoiding irrelevant benchmark-artifact downloads.

- Exactly three DGX Spark or compatible GB10 nodes
- `jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid` with the MoonViT-3d vision tower and PatchMerger projector
- The upstream `k12l1-vision` image pinned by immutable manifest digest
- Ray tensor parallelism across all three GPUs (`TP3`, `PP1`, `DCP1`)
- Hybrid NVFP4 hot experts and 2-bit AQLM cold experts
- An 11 GiB NVFP4 MLA KV allocation per rank: 354,496-token pool, 348,160-token served context
- MTP-3 speculative decoding, FULL CUDA graphs, async scheduling, and the upstream top-4 expert override
- Data-parallel multimodal encoding, required because MoonViT's 16 heads are not divisible by TP3
- NCCL 2.30.7 from NVIDIA's official ARM64 CUDA 13 wheel, SHA-256 verified and cached per node
- Persisted TorchInductor, vLLM, and FlashInfer runtime caches under SparkRun's host cache mount

Run it with the Ray head listed first:

```bash
sparkrun run @mfellner/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-348k \
  --hosts HEAD_IP,WORKER1_IP,WORKER2_IP \
  --trust \
  --no-follow
```

The recipe directly preserves MiaAI-Lab's immutable Hugging Face vision-wrapper revision `53e0082eedebd806b63e19779c47905937d768ca`.

### GLM-5.2 Vision NVFP4+AQLM — Path B, coding speed

`@mfellner/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-fp8-235k`

Path B is the coding-speed profile added by MiaAI-Lab v4.5 and retained by the latest upstream commit `5c85163ccb8d98d395880e71e2dbd03976a3f4ad`. It keeps the same immutable Vision image and checkpoint, hybrid NVFP4+AQLM weights, TP3 topology, MTP-3, FULL CUDA graphs, async scheduling, MoonViT data-parallel encoder, top-4 expert override, and validated RoCE/NCCL configuration as Path A. Its deliberate tradeoff is:

- `fp8_ds_mla` KV cache instead of `nvfp4_ds_mla`
- 12 GiB KV allocation per rank
- Approximately 240,640 KV-cache tokens
- 235,392-token served context
- `GPU_MEM_UTIL=0.9`
- Thinking disabled by default for lower-latency coding and agent use; clients can opt in per request

Upstream reports approximately 25–26 structured decode tokens/s on its three-Spark fleet, about 20% faster than Path A, in exchange for reducing context from 348,160 to 235,392 tokens. This was reproduced on the `dgx03`/`dgx04`/`gx10` ASUS Ascent GX10 cluster on 2026-08-06: after five discarded 256-token warm-ups, five deterministic 1,024-token structured-JSON runs at concurrency 1, thinking disabled, and `ignore_eos=true` delivered a **25.427 tok/s median** (25.365–25.469) with 1.063 s median client-observed TTFT. Three 512-token TypeScript coding runs delivered a **20.389 tok/s median** (20.375–20.399), within upstream's reported 15.5–21 tok/s mixed-output band. A deliberately high-entropy natural-prose prompt measured 14.236 tok/s, so decode rate remains output-dependent. A separate GPU load capture from the same acceptance session recorded busy samples in P0 at 93–96% utilization, approximately 2.44–2.50 GHz SM clocks, and 46.8–50.9 W median power; because the result JSON lacks wall-clock timestamps, this telemetry is not attributed to individual measured runs. The committed [acceptance evidence](evidence/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-fp8-235k-20260806/) includes the exact measured-workload harnesses and prompts, measured-run result JSON, raw per-node telemetry, recipe checksum, host/rank provenance, and artifact checksums.

FP8 KV can also reduce accuracy relative to Path A's NVFP4 KV cache. Benchmark application quality as well as speed before switching production traffic.

Run Path B with the Ray head listed first:

```bash
sparkrun run @mfellner/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-fp8-235k \
  --hosts HEAD_IP,WORKER1_IP,WORKER2_IP \
  --trust \
  --no-follow
```

Clients must advertise a context window of **235392** while Path B is active. For ZCode custom models, also keep `modalities.input` set to `["text", "image"]` so screenshots are not stripped.

#### Operational requirements

- Temporarily stop `earlyoom` on all three nodes before launch. Upstream reports that it can kill Ray workers during CUDA-graph capture at this memory envelope. Do not permanently disable it without operator approval; restore it only after unloading the model and confirming adequate RAM/swap headroom.
- Path B's `GPU_MEM_UTIL=0.9` startup gate requires about 109.46 GiB free per rank. When replacing another memory-saturated model, stop it first and check available memory on every node. MiaAI-Lab provides optional `DROP_CACHES` handling; SparkRun recipes cannot safely encode that host-wide operation, so if clean model file pages keep free memory below the gate, explicitly run `sync` and drop filesystem page cache on all three hosts before retrying. This affects host I/O cache and must not be automated without operator consent.
- Host order is significant: the first host is the Ray head and serves the OpenAI-compatible API on port 8000.
- The recipe caps Ray's object store at 128 MiB, matching upstream, via Ray's environment-based defaults.
- SparkRun 0.3.1 has two vllm-ray lifecycle defects that must be fixed locally (or superseded by a later release): finalize each host communication environment so `NODE_IP` becomes per-host `VLLM_HOST_IP`, and use `runtime.get_head_container_name(...)` for readiness/post-hook checks instead of assuming `_node_0`. Without the latter correction, SparkRun can report a false launch failure while the real `_head` container continues loading normally.
- On first launch, each node needs HTTPS access to `files.pythonhosted.org` to fetch the pinned NCCL 2.30.7 wheel. The recipe verifies wheel SHA-256 `ca786ffa5a647c75d4d1f5cc72a6c4f537947e2ba8823d7c8aaf768e7a7b9f77` and extracted-library SHA-256 `fc7ea66334edbc934aa25959b9907dbb2b91a1d2485beff18839afc45cbc08d0`. Subsequent launches revalidate the persistent cached library and refetch it if verification fails.
- SparkRun 0.2.40 can exit a `docker save | ssh docker load` transfer of this 35.56 GB image before registering its manifest, then continue without reporting the failed worker image. If a worker does not show image ID `sha256:6b00d3a3…` after distribution, run `docker pull ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks@sha256:f8f350d46b33858eda4f0c0a5c39a7f0b27005111d393098baaaacae18c10fdb` on that worker and rerun SparkRun. The exact pull reuses already-transferred layers.
- Cable the three CX-7 links as a cross-connected triangle: every cable must join port 0/f0 on one node to port 1/f1 on another. The recipe selects `rocep1s0f0,rocep1s0f1` for RoCE and `enP7s7` for NCCL bootstrap, matching the validated ASUS Ascent GX10/DGX Spark layout. Hosts with different interface names must adapt these three recipe values.
- The persistent runtime-cache ownership is normalized from `/cache/huggingface` rather than a fixed numeric UID, because the container user can have different UIDs across otherwise identical nodes.

#### Security

The GLM endpoint binds to `0.0.0.0:8000` without API authentication, executes pinned model remote code via `--trust-remote-code`, and supports image URLs. SparkRun also starts privileged, host-networked Ray containers on all three hosts. **Use it only on a trusted, firewalled private network; block every inference, Ray control/object-store, NCCL, and bootstrap port on every node from untrusted networks, and never expose them directly to the public Internet.** Prefer inline `data:` image payloads; if remote URLs are accepted, restrict outbound access or place an authenticated filtering proxy in front of vLLM to reduce SSRF risk.

## Validation

Validate a recipe before publishing or running it:

```bash
sparkrun recipe validate recipes/glm-5.3-flash-exl3-dflash2-dual-spark-1m.yaml
sparkrun run recipes/glm-5.3-flash-exl3-dflash2-dual-spark-1m.yaml \
  --hosts HEAD_IP,WORKER_IP \
  --dry-run --trust

sparkrun recipe validate recipes/glm-5.3-flash-exl3-dflash2-dual-spark-900k.yaml
sparkrun run recipes/glm-5.3-flash-exl3-dflash2-dual-spark-900k.yaml \
  --hosts HEAD_IP,WORKER_IP \
  --dry-run

sparkrun recipe validate recipes/deepseek-v4-flash-0731-dspark-dual-spark-1m.yaml
sparkrun run recipes/deepseek-v4-flash-0731-dspark-dual-spark-1m.yaml \
  --hosts HEAD_IP,WORKER_IP \
  --dry-run

sparkrun recipe validate recipes/deepseek-v4-flash-dual-spark-1m.yaml
sparkrun run recipes/deepseek-v4-flash-dual-spark-1m.yaml \
  --hosts HEAD_IP,WORKER_IP \
  --dry-run

sparkrun recipe validate recipes/unsloth-qwen3.6-35b-a3b-nvfp4-dgx-spark.yaml
sparkrun run recipes/unsloth-qwen3.6-35b-a3b-nvfp4-dgx-spark.yaml \
  --hosts HOST_IP \
  --dry-run

sparkrun recipe validate recipes/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-348k.yaml
sparkrun run recipes/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-348k.yaml \
  --hosts HEAD_IP,WORKER1_IP,WORKER2_IP \
  --dry-run

sparkrun recipe validate recipes/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-fp8-235k.yaml
sparkrun run recipes/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-fp8-235k.yaml \
  --hosts HEAD_IP,WORKER1_IP,WORKER2_IP \
  --dry-run
```

## Attribution

The recommended GLM 5.3 Flash EXL3 1M recipe is adapted from MiaAI-Lab's deployment repository at the exact source revision documented above. It retains the immutable EXL3 overlay image, Mia-AiLab mirror checkpoint, updated independently pinned IncoAI DFlash2 draft, TP2 native vLLM topology, fused EXL3 MoE, FP8 sparse-MLA KV, DFlash2-7, CUDA graphs, 1M context, multimodal processing, XGrammar backports, prefix caching, and parser settings. SparkRun replaces `.env`, SSH/rsync helpers, mutable tag pulls, manually ranked Docker lifecycle, and the kit-specific readiness loop with immutable distribution, per-host fabric/GID validation, native orchestration, persistent caches, and the fail-closed rank-0 gate.

The legacy GLM 5.3 Flash EXL3 900K rollback recipe remains separately pinned to the `brandonmusic/GLM-5.3-Flash-tr3-4bpw` snapshot and the older IncoAI draft revision. It preserves the previously validated 900,000-token profile and evidence; it is not presented as the current recommended upstream adaptation.

The recommended DeepSeek V4 Flash 0731 recipe is adapted from MiaAI-Lab's MIT-licensed DSpark deployment repository. SparkRun replaces Docker Compose and manually supplied node-rank lifecycle with native `vllm-distributed` orchestration while retaining the pinned Anemll image, checkpoint, DSpark MTP-5 path, NVFP4 DS-MLA KV cache, FlashInfer B12X MoE, 1M context, parser, and generation settings. The older base-checkpoint recipe remains available for compatibility and uses SparkRun's Ray-based runtime rather than the optimized 0731 DSpark stack.

The Unsloth Qwen3.6 recipe is adapted from MiaAI-Lab's MIT-licensed deployment repository. SparkRun replaces its standalone shell lifecycle, readiness loop, cache management, and direct `docker run` invocation. The adaptation retains MiaAI-Lab's purpose-built SM121 v0.26 image, mixed-quant B12X soft-fallback patch, Unsloth model, FlashInfer B12X/attention kernels, FP8 KV cache, 256K context, MTP, async scheduling, multimodal limits, reasoning, tool-calling, and generation settings.

The GLM-5.2 Vision recipes are adapted from MiaAI-Lab's MIT-licensed triple-Spark deployment repository. SparkRun replaces its `.env` files, SSH helper, manual Ray lifecycle, resource synchronization, container startup, and readiness loop. Both adaptations retain the purpose-built vision image, hybrid NVFP4+AQLM checkpoint, TP3/DCP1 topology, MTP-3, FULL CUDA graphs, MoonViT data-parallel encoder mode, tool/reasoning parsers, and kernel tuning. Path A preserves the fixed NVFP4 MLA KV allocation and 348K context; Path B preserves v4.5's FP8 MLA coding-speed profile and 235K context. SparkRun supplies model/image distribution, lifecycle tracking, persistent cache mounts, and LiteLLM discovery; both recipes explicitly select the validated two-HCA cross-connected RoCE triangle.

## License

MIT. See [LICENSE](LICENSE).
