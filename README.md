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
```

## Attribution

The DeepSeek V4 Flash recipe is adapted from MiaAI-Lab's MIT-licensed deployment repository. SparkRun replaces its Docker Compose lifecycle, custom native-multiprocessing image, and manually supplied node-rank arguments with SparkRun's compatible `vllm-node` image and Ray orchestration. The recipe retains the upstream 1M-context, FP8 KV-cache, MTP, prefix-cache, FlashInfer, reasoning, tool-calling, and generation settings.

The Unsloth Qwen3.6 recipe is adapted from MiaAI-Lab's MIT-licensed deployment repository. SparkRun replaces its standalone shell lifecycle, readiness loop, cache management, and direct `docker run` invocation. The adaptation retains MiaAI-Lab's purpose-built SM121 image, Unsloth model, FlashInfer B12X/attention kernels, FP8 KV cache, 256K context, MTP, async scheduling, multimodal limits, reasoning, tool-calling, and generation settings.

## License

MIT. See [LICENSE](LICENSE).
