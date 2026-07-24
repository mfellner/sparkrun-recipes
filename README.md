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

## Validation

Validate a recipe before publishing or running it:

```bash
sparkrun recipe validate recipes/deepseek-v4-flash-dual-spark-1m.yaml
sparkrun run recipes/deepseek-v4-flash-dual-spark-1m.yaml \
  --hosts HEAD_IP,WORKER_IP \
  --dry-run
```

## Attribution

The DeepSeek V4 Flash recipe is adapted from MiaAI-Lab's MIT-licensed deployment repository. SparkRun replaces its Docker Compose lifecycle, custom native-multiprocessing image, and manually supplied node-rank arguments with SparkRun's compatible `vllm-node` image and Ray orchestration. The recipe retains the upstream 1M-context, FP8 KV-cache, MTP, prefix-cache, FlashInfer, reasoning, tool-calling, and generation settings.

## License

MIT. See [LICENSE](LICENSE).
