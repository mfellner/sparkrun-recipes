# mfellner SparkRun Recipes

Custom [SparkRun](https://sparkrun.dev/) recipe registry for NVIDIA DGX Spark and compatible GB10 systems.

## Add the registry

```bash
sparkrun registry add --trust https://github.com/mfellner/sparkrun-recipes.git
```

Recipes are then available under the `@mfellner/` namespace. Review the repository before granting trust: several recipes intentionally require custom images, host networking/IPC, device access, capabilities, or lifecycle hooks. If the registry was already added without trust, run `sparkrun registry trust mfellner` after auditing it, or add `--trust` explicitly to a local-file launch.

## Recipes

### DeepSeek V4 Flash 0731 DSpark — dual Spark, 1M context (recommended)

`@mfellner/deepseek-v4-flash-0731-dspark-dual-spark-1m`

An immutable SparkRun adaptation of [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark). The effective serving stack was audited against upstream commit `a4ce87a2f47f1be8fe64c297a0cf33a9a5e509aa`.

- Two DGX Spark or compatible GB10 nodes using SparkRun's native `vllm-distributed` runtime
- `deepseek-ai/DeepSeek-V4-Flash-0731` at revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Anemll DSpark image `0.1.1`, pinned at manifest digest `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
- Tensor parallelism 2, 1,048,576-token configured context, and NVFP4 DS-MLA KV cache
- Native DSpark speculative decoding with five probabilistically sampled draft tokens
- FlashInfer B12X MXFP4 MoE, SM121 kernels, async scheduling, chunked prefill, prefix caching, and FlashInfer autotuning
- DeepSeek V4 reasoning and tool-call parsers, with reasoning effort `max` by default as in the pinned upstream revision; clients can disable thinking per request for low-latency decode benchmarking
- Encoder compatibility patch copied from the pinned model snapshot only after SHA-256 verification
- 64 GiB shared memory, persistent runtime/autotune caches, RoCE v2/NCCL environment, and readiness plus semantic-completion post-launch gates

Port `8000` is an intentional operational adaptation from upstream's `8888`, preserving SparkRun registry discovery and existing proxy conventions. It does not change model execution or benchmark methodology. The served model name remains upstream-compatible: `deepseek-v4-flash-0731`.

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

#### Reproduced performance

Acceptance testing on two ASUS Ascent GX10 systems used 2,048-token structured-output streams, temperature 0, `ignore_eos=true`, and thinking disabled. Five warmed single-stream HTML runs produced:

- median TTFT: **239.5 ms** (range **182.8–280.9 ms**)
- median streamed decode: **81.96 tok/s** (range **81.62–83.39 tok/s**)
- median end-to-end output rate: **81.27 tok/s**

This reproduces MiaAI-Lab's published `80+ tok/s` performance class; their screenshot reports 82.4 tok/s for its selected structured-output stream. Decode performance is output-dependent because DSpark acceptance varies with generated content: the same local four-prompt suite ranged from 70.78 to 81.71 tok/s at concurrency 1. A warmed thinking-disabled concurrency sweep measured **101.32 aggregate tok/s at C2** and **142.25 aggregate tok/s at C4**. Keep decode TPS, TTFT, end-to-end output rate, and aggregate concurrent throughput separate when comparing results.

The configured 1,048,576-token limit and approximately 1.9M-token KV pool are configured capabilities. Empirical semantic validation reached 62,032 prompt tokens and returned the requested sentinel; a near-1M prompt was not part of this acceptance run.

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

The recommended DeepSeek V4 Flash 0731 recipe is adapted from MiaAI-Lab's MIT-licensed DSpark deployment repository. SparkRun replaces Docker Compose and manually supplied node-rank lifecycle with native `vllm-distributed` orchestration while retaining the pinned Anemll image, checkpoint, DSpark MTP-5 path, NVFP4 DS-MLA KV cache, FlashInfer B12X MoE, 1M context, parser, and generation settings. The older base-checkpoint recipe remains available for compatibility and uses SparkRun's Ray-based runtime rather than the optimized 0731 DSpark stack.

The Unsloth Qwen3.6 recipe is adapted from MiaAI-Lab's MIT-licensed deployment repository. SparkRun replaces its standalone shell lifecycle, readiness loop, cache management, and direct `docker run` invocation. The adaptation retains MiaAI-Lab's purpose-built SM121 v0.26 image, mixed-quant B12X soft-fallback patch, Unsloth model, FlashInfer B12X/attention kernels, FP8 KV cache, 256K context, MTP, async scheduling, multimodal limits, reasoning, tool-calling, and generation settings.

The GLM-5.2 Vision recipes are adapted from MiaAI-Lab's MIT-licensed triple-Spark deployment repository. SparkRun replaces its `.env` files, SSH helper, manual Ray lifecycle, resource synchronization, container startup, and readiness loop. Both adaptations retain the purpose-built vision image, hybrid NVFP4+AQLM checkpoint, TP3/DCP1 topology, MTP-3, FULL CUDA graphs, MoonViT data-parallel encoder mode, tool/reasoning parsers, and kernel tuning. Path A preserves the fixed NVFP4 MLA KV allocation and 348K context; Path B preserves v4.5's FP8 MLA coding-speed profile and 235K context. SparkRun supplies model/image distribution, lifecycle tracking, persistent cache mounts, and LiteLLM discovery; both recipes explicitly select the validated two-HCA cross-connected RoCE triangle.

## License

MIT. See [LICENSE](LICENSE).
