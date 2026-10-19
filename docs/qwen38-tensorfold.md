# Qwen3.8-27B on one Spark: TensorFold

`@mfellner/qwen3.8-27b-tensorfold-dgx-spark`

Adapted from MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold at
`c11ff138871cf43d12898d79dbfa4dc952d17ca4`. TensorFold v0.6.0 is the actual
engine; SparkRun 0.4.1's `vllm-distributed` runtime is only the solo
Docker/lifecycle shim, not the inference engine. Use SparkRun 0.4.1 or later
for this recipe; versions later than 0.4.1 have not been live-tested here.
The public ARM64 image is pinned to
`sha256:043d192b9e1725f55c5439fc9672055de891433e5b31448ed8963918e3c460e9`,
with `tf.patches=fb91a0019fdc` (Mia's five patches, no engine modifications here).
Target and drafter snapshots are pinned independently. The target uses SparkRun
distribution and its validated `{model}` binding, which resolves to the exact
prepared snapshot before serving on 0.4.1. The drafter path is relative to the
managed `$HF_HUB_CACHE` mount. A checksum-pinned pre-launch mod stages the drafter.
This mod originated as a workaround for SparkRun 0.3.6's local per-entry revision
bug (the old failed launch receipt is retained); it remains an explicit integrity
check, not a claim that 0.4.1 still has that bug.
The mod first hashes the cached draft config and 3.85 GB safetensor
against reviewed constants; on missing/corrupt data it downloads only those files
at the exact draft revision, without forwarding a token, then rechecks fail-closed.
No update of a moving Hub ref or `trust_remote_code` is required. A fully verified
cache needs no Hub contact from the drafter mod; missing data requires network
access during preparation, before the offline server starts.

## Profile and intentional translations

- Vontra/Qwen3.8-27B-MLX-4bit at `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`.
- z-lab/Qwen3.8-27B-DFlash2 at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- Native 262,144 prompt-plus-reply tokens, 8 streams; YaRN off.
- FP8 KV cache and FP8 prompt activations. The latter trades numerical prompt
  precision for speed; this is upstream's default, not a local quality finding.
- Thinking on; sampling defaults temperature 1.0, top_p 0.95, top_k 20.
- Inline images/video enabled; remote URL fetching disabled. Limits are upstream's
  50 images and 16,384 image/video tokens, not local exhaustive media qualification.
- A fixed 78 GiB pool selects upstream auto's maximum on an idle 128 GB host.
  Unlike auto, it refuses inadequate memory instead of silently reducing capacity.
  Require at least 109 GiB MemAvailable and inactive earlyoom before launch.
- Port 8000 replaces upstream 8888. Loopback replaces wildcard binding by default.
  For LAN clients set `-o host=YOUR_PRIVATE_LAN_IP` explicitly.
- `mods/qwen38-tensorfold/serve.py` converts the one SparkRun
  `--served-model-name` argument into TensorFold's `--name`, rejecting a duplicate,
  missing value, or ambiguous native alias. TensorFold validates all other flags.
- SparkRun mounts its existing HF cache. Compilation caches now use SparkRun's
  managed `/cache/runtime` mount, keyed by both model/revision and image identity.
  Recipe-local HF_HOME/TORCH_EXTENSIONS_DIR/TRITON_CACHE_DIR overrides and the
  mkdir hook have been removed; old cache directories are not deleted. Automatic
  pruning is disabled for this recipe to avoid deleting other runtime-cache trees.
  Persistence configuration alone is not proof of measured reuse.

## Launch and stop

Review this repository before trusting it:

```bash
sparkrun registry add --trust https://github.com/mfellner/sparkrun-recipes.git
# If already registered:
sparkrun registry update mfellner
```

Check free disk (allow approximately 60 GB for image plus checkpoints), available
RAM, GPU workloads and listeners first. SparkRun 0.3.6's non-delegating launcher
also attempts a host-wide Linux page-cache drop when its scoped sudo permission
exists; the accepted run's kernel receipt records this. Review that I/O-cache
side effect before running on a shared host. On the target host the operator runs:

```bash
sudo systemctl stop earlyoom
systemctl is-active earlyoom  # expected inactive
free -h
```

This is temporary, not `disable`. Then:

```bash
sparkrun run @mfellner/qwen3.8-27b-tensorfold-dgx-spark \
  --hosts YOUR_SPARK_LAN_IP --no-follow
curl -fsS http://127.0.0.1:8000/v1/models
```

For a LAN-only bind on SparkRun 0.4.1, add
`-o host=YOUR_PRIVATE_LAN_IP --no-ready-wait` and perform the HTTP and semantic
checks yourself against that LAN IP. SparkRun's local readiness observer probes
`127.0.0.1` even when the engine binds only the LAN address: an initial LAN-bound
attempt served successfully, but the CLI wait timed out and is **not** accepted
as a successful launch receipt. Do not widen the bind to all interfaces merely
to make that observer pass. The default loopback recipe avoids this limitation.

Do not infer readiness from launch completion. Wait for `/v1/models`, inspect
`/health`, and require a semantic chat completion. First startup can compile CUDA
kernels. Do not restart a healthy progressing compiler. The API alias is
`Qwen3.8-27B`; use adequate max_tokens for thinking, or request
`"chat_template_kwargs":{"enable_thinking":false}` for exact smoke tests.

Stop using the exact SparkRun workload ID printed at launch:

```bash
sparkrun stop WORKLOAD_ID
# Only after confirming the model has unloaded and memory headroom is safe:
sudo systemctl start earlyoom
```

SparkRun's supervisor is not an automatic engine-process or host-reboot recovery
policy. Never use `sparkrun stop --dry-run` on 0.3.6: it can actually stop workloads.

## Security and licensing

The inference API has no authentication. The recipe uses a root container user,
host networking/IPC, IPC_LOCK and GPU access; it explicitly sets Docker privileged
false, no-new-privileges and no RDMA device mounts. This is a trusted-host/private-
LAN profile, not an internet deployment. The cache is writable from the container.
Keep remote URL media disabled, and add authentication/ingress policy before
exposing the API beyond trusted clients. Inspect effective Docker state after
upgrading SparkRun; YAML intent alone is not proof of executor behavior.

Mia's launcher/patches and TensorFold v0.6.0 are Apache-2.0, with MIT notices for
older/adapted parts in the upstream tree. The selected target's pinned card and
bundled LICENSE report Apache-2.0; the draft card also declares Apache-2.0 but has
no standalone LICENSE. Mia's README instead calls the weights Qwen Community
License 1.0. Preserve this publisher-prose discrepancy rather than treating its
summary or the image's Apache OCI label as an override of artifact-specific terms.
Review the pinned artifacts, upstream NOTICE and NVIDIA PyTorch container terms
before relying on a commercial-use grant. Newly authored alias adapter code follows
this repository's license. No upstream source code is vendored by the adapter.

## Acceptance boundary

The configuration above is not a claim of locally measured 262K retrieval,
8 full simultaneous windows, 50-image/video validation, upstream throughput,
quality parity, or long-term uptime. The [historical 0.3.6 LAN-process evidence](../evidence/qwen38-tensorfold-20261019/)
passes direct chat, eight overlapping clients, image understanding, typed tools,
41,443-token retrieval, proxy chat/image/tools and process/log checks for the
**old recipe at commit `01284da`**, not the updated runtime-cache configuration.
Its exact-recipe verifier must be replayed from that Git revision. Updated
SparkRun 0.4.1 evidence is separate in
[qwen38-tensorfold-sparkrun041-20261019](../evidence/qwen38-tensorfold-sparkrun041-20261019/).
No old process receipt is reused to qualify a new process or recipe.

SparkRun 0.4.1's validator still emits `unpinned-model-revision`: that check
assumes a vLLM engine needs `--revision` and does not recognize the prepared
`{model}` snapshot binding. TensorFold does not accept that vLLM flag. Keep the
pin and verify the actual listener argv and SparkRun prepared-model metadata;
do not drop the pin or add an unsupported flag to silence this false positive.
