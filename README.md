# mfellner SparkRun Recipes

Custom [SparkRun](https://sparkrun.dev/) recipe registry for NVIDIA DGX Spark and compatible GB10 systems.

## Add the registry

```bash
sparkrun registry add https://github.com/mfellner/sparkrun-recipes.git
```

Recipes are then available under the `@mfellner/` namespace.

## Recipes

### DeepSeek V4 Flash — dual Spark, 1M context

`@mfellner/deepseek-v4-flash-dual-spark-1m`

A SparkRun adaptation of [MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context](https://github.com/MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context). It uses:

- Two DGX Spark or compatible GB10 nodes
- SparkRun's proven `vllm-ray` multi-node orchestration
- Tensor parallelism across both GPUs
- 1,000,000-token configured context
- FP8 KV cache
- MTP speculative decoding
- Prefix caching and FlashInfer autotuning
- DeepSeek V4 reasoning and tool-call parsers
- The SparkRun-compatible local `vllm-node` image, while retaining MiaAI-Lab's model, FP8 KV-cache, 1M-context, MTP, and FlashInfer tuning

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

### Unsloth Qwen3.6 35B A3B NVFP4 — single Spark, 256K context

`@mfellner/unsloth-qwen3.6-35b-a3b-nvfp4-dgx-spark`

A SparkRun adaptation of [MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark](https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark). It uses:

- One DGX Spark or compatible GB10 node
- `unsloth/Qwen3.6-35B-A3B-NVFP4`
- MiaAI-Lab's purpose-built vLLM 0.24.1-dev / FlashInfer image
- SM121-native FlashInfer B12X linear kernels
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

The recipe clears the image's standalone entrypoint so SparkRun controls the container lifecycle, while retaining the upstream image, model, kernels, and vLLM flags. It also aligns the image's baked Docker healthcheck with the actual served model ID and persists FlashInfer/vLLM compilation caches through SparkRun's host cache mount, avoiding a full kernel rebuild after normal recipe restarts.

#### Security

This upstream-compatible recipe binds vLLM to `0.0.0.0`, does not configure API authentication, and permits URL-based vision input from any media domain. **Run it only on a trusted, firewalled private network. Do not expose port 8000 to the public Internet.** In untrusted or multi-tenant environments, restrict ingress with a host firewall or reverse proxy, add authentication, and replace `--allowed-media-domains '*'` with an explicit allowlist before deployment.

### GLM-5.2 Vision NVFP4+AQLM — Path A, maximum context

`@mfellner/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-348k`

A SparkRun adaptation of [MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks). It uses:

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
  --no-follow
```

The recipe directly preserves MiaAI-Lab's immutable Hugging Face vision-wrapper revision `53e0082eedebd806b63e19779c47905937d768ca`.

### GLM-5.2 Vision NVFP4+AQLM — Path B, coding speed

`@mfellner/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-fp8-235k`

Path B is the coding-speed profile added by MiaAI-Lab v4.5 at upstream commit `893d869ecdf002f127f44da66b71d53b157254aa`. It keeps the same immutable Vision image and checkpoint, hybrid NVFP4+AQLM weights, TP3 topology, MTP-3, FULL CUDA graphs, async scheduling, MoonViT data-parallel encoder, top-4 expert override, and validated RoCE/NCCL configuration as Path A. Its deliberate tradeoff is:

- `fp8_ds_mla` KV cache instead of `nvfp4_ds_mla`
- 12 GiB KV allocation per rank
- Approximately 240,640 KV-cache tokens
- 235,392-token served context
- `GPU_MEM_UTIL=0.9`
- Thinking disabled by default for lower-latency coding and agent use; clients can opt in per request

Upstream reports approximately 25–26 structured decode tokens/s on its three-Spark fleet, about 20% faster than Path A, in exchange for reducing context from 348,160 to 235,392 tokens. Treat those values as upstream expectations until reproduced on the target cluster.

FP8 KV can also reduce accuracy relative to Path A's NVFP4 KV cache. Benchmark application quality as well as speed before switching production traffic.

Run Path B with the Ray head listed first:

```bash
sparkrun run @mfellner/glm-5.2-nvfp4-aqlm-triple-dgx-spark-vision-fp8-235k \
  --hosts HEAD_IP,WORKER1_IP,WORKER2_IP \
  --no-follow
```

Clients must advertise a context window of **235392** while Path B is active. For ZCode custom models, also keep `modalities.input` set to `["text", "image"]` so screenshots are not stripped.

#### Operational requirements

- Temporarily stop `earlyoom` on all three nodes before launch. Upstream reports that it can kill Ray workers during CUDA-graph capture at this memory envelope. Do not permanently disable it without operator approval; restore it only after unloading the model and confirming adequate RAM/swap headroom.
- Path B's `GPU_MEM_UTIL=0.9` startup gate requires about 109.46 GiB free per rank. When replacing another memory-saturated model, stop it first and check available memory on every node. MiaAI-Lab provides optional `DROP_CACHES` handling; SparkRun recipes cannot safely encode that host-wide operation, so if clean model file pages keep free memory below the gate, explicitly run `sync` and drop filesystem page cache on all three hosts before retrying. This affects host I/O cache and must not be automated without operator consent.
- Host order is significant: the first host is the Ray head and serves the OpenAI-compatible API on port 8000.
- The recipe caps Ray's object store at 128 MiB, matching upstream, via Ray's environment-based defaults.
- On first launch, each node needs HTTPS access to `files.pythonhosted.org` to fetch the pinned NCCL 2.30.7 wheel. The recipe verifies wheel SHA-256 `ca786ffa5a647c75d4d1f5cc72a6c4f537947e2ba8823d7c8aaf768e7a7b9f77` and extracted-library SHA-256 `fc7ea66334edbc934aa25959b9907dbb2b91a1d2485beff18839afc45cbc08d0`. Subsequent launches revalidate the persistent cached library and refetch it if verification fails.
- SparkRun 0.2.40 can exit a `docker save | ssh docker load` transfer of this 35.56 GB image before registering its manifest, then continue without reporting the failed worker image. If a worker does not show image ID `sha256:6b00d3a3…` after distribution, run `docker pull ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks@sha256:f8f350d46b33858eda4f0c0a5c39a7f0b27005111d393098baaaacae18c10fdb` on that worker and rerun SparkRun. The exact pull reuses already-transferred layers.
- Cable the three CX-7 links as a cross-connected triangle: every cable must join port 0/f0 on one node to port 1/f1 on another. The recipe selects `rocep1s0f0,rocep1s0f1` for RoCE and `enP7s7` for NCCL bootstrap, matching the validated ASUS Ascent GX10/DGX Spark layout. Hosts with different interface names must adapt these three recipe values.
- The persistent runtime-cache ownership is normalized from `/cache/huggingface` rather than a fixed numeric UID, because the container user can have different UIDs across otherwise identical nodes.

#### Security

The GLM endpoint binds to `0.0.0.0:8000` without API authentication, executes pinned model remote code via `--trust-remote-code`, and supports image URLs. SparkRun also starts privileged, host-networked Ray containers on all three hosts. **Use it only on a trusted, firewalled private network; block every inference, Ray control/object-store, NCCL, and bootstrap port on every node from untrusted networks, and never expose them directly to the public Internet.** Prefer inline `data:` image payloads; if remote URLs are accepted, restrict outbound access or place an authenticated filtering proxy in front of vLLM to reduce SSRF risk.

## Validation

Validate a recipe before publishing or running it:

```bash
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

The DeepSeek V4 Flash recipe is adapted from MiaAI-Lab's MIT-licensed deployment repository. SparkRun replaces its Docker Compose lifecycle, custom native-multiprocessing image, and manually supplied node-rank arguments with SparkRun's compatible `vllm-node` image and Ray orchestration. The recipe retains the upstream 1M-context, FP8 KV-cache, MTP, prefix-cache, FlashInfer, reasoning, tool-calling, and generation settings.

The Unsloth Qwen3.6 recipe is adapted from MiaAI-Lab's MIT-licensed deployment repository. SparkRun replaces its standalone shell lifecycle, readiness loop, cache management, and direct `docker run` invocation. The adaptation retains MiaAI-Lab's purpose-built SM121 image, Unsloth model, FlashInfer B12X/attention kernels, FP8 KV cache, 256K context, MTP, async scheduling, multimodal limits, reasoning, tool-calling, and generation settings.

The GLM-5.2 Vision recipes are adapted from MiaAI-Lab's MIT-licensed triple-Spark deployment repository. SparkRun replaces its `.env` files, SSH helper, manual Ray lifecycle, resource synchronization, container startup, and readiness loop. Both adaptations retain the purpose-built vision image, hybrid NVFP4+AQLM checkpoint, TP3/DCP1 topology, MTP-3, FULL CUDA graphs, MoonViT data-parallel encoder mode, tool/reasoning parsers, and kernel tuning. Path A preserves the fixed NVFP4 MLA KV allocation and 348K context; Path B preserves v4.5's FP8 MLA coding-speed profile and 235K context. SparkRun supplies model/image distribution, lifecycle tracking, persistent cache mounts, and LiteLLM discovery; both recipes explicitly select the validated two-HCA cross-connected RoCE triangle.

## License

MIT. See [LICENSE](LICENSE).
