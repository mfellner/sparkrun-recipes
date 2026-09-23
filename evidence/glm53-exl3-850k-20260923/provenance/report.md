# GLM-5.3 EXL3 upstream delta / provenance audit

## Verdict

**The latest TP2 default execution profile does not require rebuilding the optional thin-decode extension.** Pin public `:exl3-instanttensor` at `sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d`, install the exact latest Python runtime overlays, and explicitly keep `GLM53_EXL3_MOE_FAST=0`. New `glm53_fast_moe_version()==1` is required only with FAST=1. Keeping FAST=0 is upstream default behavior, not a performance deviation.

This is **default-profile compatibility**, not an assertion that the old image is a byte-identical build of latest Dockerfile or supports every new opt-in. Latest Dockerfile adds the native fast path unconditionally at build time. An exact latest-image build or the new long-coding example with FAST=1 requires a rebuild and GPU qualification. No GPU qualification or refreshed model launch was performed here.

## Scope and evidence

- Reviewed upstream `f906ee990596486e10ddbe381efa6f0e496f77e3` → `3f2be18c41effca0b4b2c6a65f0b24a7a9f39567` (remote HEAD rechecked at audit end).
- Upstream: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
- Read-only source tree: `/home/max/.hermes/cache/scratch/glm53-upstream-20260923`.
- Local recipe baseline: repository `4f8ecea3a513a151d772de1e067ebcbd72b2f684`; `recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml` SHA256 `03b5fe957c71632463dbd7b982d6b234251a6da7f2d18af6c9316a84dcaaf614`.
- Net delta: 187 files, 43,196 insertions, 753 deletions. Full mechanical file inventory is `changed-files.tsv`; complete diff is `delta.diff`; merged history is `commits.txt`. Use the net diff, not every merged side-branch commit subject, to infer behavior changes.
- Raw anonymous GHCR manifests/configs, Hugging Face revision/file maps, PyPI metadata, CPU-test receipt, and source/default inventories are adjacent to this report.
- Only scratch evidence was written by this audit. No Docker commands, model requests, GPU imports, builds, pulls, service changes, or remote-host mutations were made.
- The parent was concurrently updating the local recipe repository. `audit-scope.json` records its worktree status; those changes are not this audit's edits. The audited YAML itself was unchanged at the final checksum check. This is an upstream/baseline audit, not approval of the parent's evolving final diff.

## 1. Required TP2 adaptation: material runtime deltas

| Area | Net change since f906ee9 | Required treatment |
|---|---|---|
| Image / loader | Default tag `exl3` → `exl3-instanttensor`; InstantTensor 0.2.0 direct-I/O safetensors loader | Pin new digest and pass `--load-format instanttensor` explicitly. Digest strings lack `instanttensor`, so do not rely on upstream's image-name heuristic. |
| Explicit KV reservation | Fresh `.env.example` now supplies `EXTRA_ARGS="--kv-cache-memory-bytes 15032385536"` (14 GiB) | Include explicit CLI reservation at 850000 / 0.85 with InstantTensor. This is a dotenv default, **not** a hardcoded fallback added to all existing `.env` files. Preflight merely warns if missing. Upstream measured automatic pool too small for 850K. |
| Fair mixed-prefill | `GLM53_MIXED_PREFILL_CHUNK=skip` → `fair`; scheduler v5 service-time policy | Update environment and `patch_scheduler_decode_floor.py` together; fair settings below. Existing `skip` is no longer latest default fidelity. |
| Scheduler restart correctness | Position-independent helper verification with exact helper-region validation | Include current scheduler patch and restart regression test; existing adaptive-k insertion must not make a valid restart fail. |
| Mamba align chunking | New `patch_mamba_align_chunking.py` uses Mamba-group block size, not drafter-lowered cache block size; EAGLE back-off only for full-attention EAGLE group | Apply immediately after scheduler v5, before drafter/coordinator changes. Fix is required even with compact draft pages OFF. |
| Mamba state lifetime / budget | New `patch_mamba_align_state_free.py` retains a bounded list of superseded state indices instead of overwriting one tracker; frees with original predicate; align reservation includes `1 + max_concurrent_batches + num_speculative_blocks` (10 instead of 9 here) | Apply current patch on both ranks after coordinator/no-store overlays. Changes actual running-state budget; do not retain old 883552-token capacity claim. Current 14 GiB documentation says 876958 tokens. |
| Prefix coordinator correctness | `patch_hybrid_prefix_hit.py` now migrates supported stock-image legacy forms; validates complete owned stages/helper bindings, duplicate/stale/edited stages, AST/compiler binding ownership, and writes fail-closed | Use latest exact bytes, not just add compact-mode flag. New stricter composition checks apply even with compact OFF. |
| Compact drafter plumbing | `patch_glm5_drafter_group.py` adds optional page-fitting block derivation, DFlash-only grouping preflight, explicit-stride split rejection and boundary lookup support; `patch_hybrid_prefix_hit.py` changes lookup only under opt-in | Install code, explicitly keep `GLM53_DRAFT_KV_COMPACT=0` for default profile. No default compact behavior claim. |
| Per-request APC no-store | New `patch_apc_no_store.py` supports client `vllm_xargs.skip_writing_prefix_cache`; cache reads remain possible | Set `GLM53_APC_NO_STORE=1` and apply after hybrid + per-group retention. Request flag defaults OFF; server support defaults ON. Include mandatory composed real-vLLM tests separately from launch. |
| KV capacity diagnostics | New `patch_kv_capacity_log.py`, current Mamba/compact accounting | Set `GLM53_KV_CAPACITY_LOG=1`; apply after drafter/coordinator changes. Log usable IDs, per-group allocation, running-request vs cached-history costs, not an invented single pool-token count. |
| Omitted output limits | New `patch_default_max_new_tokens.py` | `DEFAULT_MAX_NEW_TOKENS=65536`. Omitted-only fallback for chat/completions; explicit client limits still win; not an admission or hard server cap. Empty disables override. Patch protocol normalization so omitted legacy completion default is not confused with explicit 16. |
| Tool control | New `patch_tool_choice_none.py` and preflight | Enforce `tool_choice:none` at decode via glm47 parser hook. Include static + semantic tests. This is separate from existing auto-tool-choice setup. |
| Vision memory controls | Upstream limit 100 images / 1 video → 48 / 1; new per-image token cap and processor-cache size | Add `--mm-processor-kwargs '{"max_image_tokens":2048}'` and `--mm-processor-cache-gb 1`. Existing local image-only 4/0 and media.invalid boundary can remain intentional stricter policy. Never increase accepted media merely because upstream does. |
| Warmup | Adds long-prefill rungs 3584, 7168, 14336, 65536 to existing DFlash BLOCK/sampler warmup; linear prompt generation and file-body curl requests avoid quadratic building / ARG_MAX | Refresh warmup bytes and acceptance counting. Parent's stricter fail-closed readiness supervision may remain. |
| Adaptive-k robustness | Empty/whitespace environment values now normalize to defaults in policy and capture helpers | Vendor updated patch; default mode still `off`. |
| EXL3 Python module | Optional fast-decode gate/pointer alias, KDA BF16 large-M path, TP3 Marlin-alignment exceptions | Install exact current `overlay/exl3.py`; keep new options explicitly OFF for standard TP2. Existing E3 kernel path remains valid on public image. |
| Cache reset support | New `patch_cache_reset.py` for cache-only dev routes | Install support if claiming full launcher-overlay fidelity, but keep `GLM53_EXPOSE_CACHE_RESET=0`. These root-mounted routes are outside bearer guard; do not enable for ordinary serving. |

The chat template itself has **no net change** in this comparison. Neither do the existing native E2/E3 source files (`exl3_fat_gemm.*`, `exl3_fat_moe.*`) or their patcher. Do not invent a required E3 kernel rebuild from unrelated commit-log history.

### Exact current TP2 runtime patch order

From `start.sh` `GLM53_OVERLAY_ORDER`:

1. `patch_glm_video_placeholders.py`
2. `patch_suppress_stops_in_reasoning.py`
3. `patch_scheduler_decode_floor.py`
4. `patch_mamba_align_chunking.py`
5. `patch_glm5_drafter_group.py`
6. `patch_hybrid_prefix_hit.py`
7. `patch_apc_per_group_retention.py`
8. `patch_apc_no_store.py`
9. `patch_mamba_align_state_free.py`
10. `patch_kv_capacity_log.py`
11. `patch_tool_choice_none.py`
12. `patch_xgrammar_termination.py`
13. `patch_kpool_tail_slotmap.py`
14. `patch_spinwait.py`
15. `patch_adaptive_k.py`
16. `patch_dense_fp8.py`
17. `patch_default_max_new_tokens.py`
18. `patch_indexer_workspace.py`
19. `patch_cache_reset.py`
20. `patch_ablit.py`

Local multi-token reasoning-stop fix / E3 instrumentation are deliberate local correctness/evidence adaptations, not upstream bytes. Preserve and separately test their composition rather than silently replacing them. Dockerfile build order also contains prerequisites such as model overrides / DFlash / Eagle setup; do not confuse this runtime array with a clean-image build recipe.

## 2. Current launcher defaults (not the optional example)

### Core serving profile

- TP=2, NNODES=2, native multiprocessing (`mp`), rank-1 headless; pipeline parallel remains 1 in local recipe.
- Context 850000; utilization 0.85; max sequences 4; batched tokens 7168.
- Quantization `exl3`; target KV `fp8`; DFlash k=7; draft TP=2; draft KV auto; probabilistic draft / standard rejection.
- CUDA graph capture sizes `1 2 4 8 16 24 32`; eager OFF; prefix caching ON; FlashInfer autotune disabled.
- Tool parser `glm47`, auto-tool-choice enabled; reasoning parser `glm45`; `/opt/glm53/chat_template.jinja`.
- Port 8888 upstream; local 8000 is deliberate endpoint translation.
- E3: `EXL3_FUSED_MOE=1`, `EXL3_MOE_ROW_TILE=0`, `EXL3_TEMP_ROWS_FUSED=32`, `EXL3_FAT_SORTED=0`, `EXL3_FAT_BATCHED=0`, `EXL3_FAT_KERNEL=1`, `EXL3_FAT_GROUPED=1`. Higher tiers imply lower internals; do not reinterpret the literal zero flags as disabling E3.
- `LOAD_FORMAT=instanttensor` in fresh `.env.example`; fallback `start.sh` chooses it only if IMAGE contains `instanttensor`. Explicit empty means vLLM auto. With a digest-only pinned image, set the serve flag explicitly.
- Fresh dotenv KV cap `15032385536` bytes; existing customized dotenv may have none. This distinction matters for provenance.

### Policy defaults / new switches

| Setting | Latest value |
|---|---|
| `GLM53_MIXED_PREFILL_CHUNK` | `fair` |
| `GLM53_FAIR_PREFILL_CHUNK` | `256` |
| `GLM53_FAIR_PREFILL_SHARE` | `0.30` |
| `GLM53_FAIR_PREFILL_MAX_INTERVAL_MS` | `2000` |
| `GLM53_FAIR_PREFILL_MAX_STEP_MS` | `2000` |
| `GLM53_FAIR_PREFILL_MAX_CHUNKS` | `1` |
| `DEFAULT_MAX_NEW_TOKENS` | `65536` |
| `GLM53_APC_NO_STORE` / `GLM53_KV_CAPACITY_LOG` | `1` / `1` |
| `GLM53_EXL3_MOE_FAST` / `GLM53_KDA_BF16_LARGE_M` | `0` / `0` |
| `GLM53_DRAFT_KV_COMPACT` | `0` |
| `GLM53_EXPOSE_CACHE_RESET` | `0` |
| `GLM53_INDEXER_WORKSPACE` | `rightsize` |
| `GLM53_SPINWAIT_MS` | `stock` (not 16) |
| `GLM53_ADAPTIVE_K` / `GLM53_DENSE_FP8` | `off` / `off` |
| `GLM53_ADAPTIVE_K_SET/ALPHA/MARGIN/MIN_STEPS/SATURATE/HIST` | `2,4,7` / `0.25` / `1.0` / `4` / `max` / `200` |
| `GLM53_SUPPRESS_STOPS_IN_REASONING` | `1` |
| `GLM53_DEFAULT_REASONING_EFFORT` | empty; template fallback unchanged |
| Global/SWA APC retention | unset/empty: inherit stock policy; no automatic sparse retention |
| `LONG_PREFILL_TOKEN_THRESHOLD` / `PREFIX_MATCH_UNIT` | empty / empty; flags omitted |
| `NCCL_NCHANNELS` | empty; **8 is not a default**. Nonempty value pins MIN and MAX channels equally. |
| `USE_HOST_NCCL` | `0`; no host preload by default |
| `CG_ESTIMATE` | `1` |
| `ABLIT` | `0`; now forcibly reset after dotenv, explicit caller export may opt in |
| `GLM53_COOP_GEOMETRY` / overlay selection | empty / stock overlay; cooperative runtime not selected |
| `NFS_SHARE` | `0` |
| `LANGUAGE_MODEL_ONLY` / `SKIP_MM_PROFILING` | `0` / `1` |
| `LIMIT_MM` | `{"image":48,"video":1}` |
| `MM_IMAGE_TOKENS` / `MM_PROCESSOR_CACHE_GB` | `2048` / `1` |
| `VIDEO_NUM_FRAMES` | empty; underlying vLLM default 32, not explicitly requested by launcher |
| Engine execution timeout / readiness timeout | `1800` / `3600` seconds |
| Boot shape warmup / request timeout | `1` / `240` seconds |

Raw top-level assignment inventories are `launcher-old-assignments.txt` and `launcher-new-assignments.txt`; these deliberately do not source or execute start.sh.

## 3. Image and compiled-extension provenance

Registry repository: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks`.

| Tag | Anonymous manifest digest | Created (config) |
|---|---|---|
| `exl3` | `sha256:eecb36e14dc34c92d46827fde7b09f7e0bf27e27c426ece126376c02dea6cd2f` | 2026-09-07T08:54:38.100423599+03:00 |
| `exl3-instanttensor` | `sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d` | 2026-09-16T10:12:01.210903564+03:00 |

Both resolve directly to arm64 image manifests (no multiarch child-selection ambiguity). Image config has `Entrypoint=["vllm","serve"]`, workdir `/vllm-workspace`, unspecified user (Docker default root), no declared healthcheck. OCI source/version/revision labels are inherited vLLM placeholders (`source=https://github.com/vllm-project/vllm`, revision `unknown`, version `local/vllm-openai:dev`), **not** proof of exact GLM launcher source. InstantTensor image recipe stamp is `5fd1b44ac47e04e8c72497752bf83b3c4b48e8e6cd20e8e0ece4ef6001161aea`.

The new public tag is a Sept-16 wheel addition over a Sept-11 runtime. Both public tags share exactly the same Sept-7 compiled EXL3 layer:

`sha256:a0eeaa2c3ad77bfd0576a02cae62d48fd20ded5c089e90d89a20efc669a595a4` (compressed size 42,947,920 bytes).

History proves that layer applies aarch64 and E2/E3 patchers and verifies existing symbols, but not `patch_exl3_decode_pipeline.py` or `glm53_fast_moe_version`. No later history entry rebuilds that extension. This was metadata/layer-identity inspection, not an import of the live serving binary.

### Why FAST=0 is compatible

- `overlay/exl3.py:1250–1278` validates 0/1 and checks the new symbol only under `if fast:`.
- Gate/up SUH pointer alias is also conditional on fast mode; OFF preserves separate stock pointer tables.
- Latest native patch only adds K4/N256 SM121 fast kernels and guarded host dispatch; the existing E2/E3 source files are unchanged.
- Latest `tests/test_exl3_decode_pipeline.py` ran CPU-only: **13 tests passed** (`thin-decode-cpu-tests.log`). It uses real-source AST extraction with fake tensors, not CUDA.
- Additional real-source CPU check built fused state using an extension stub **without** `glm53_fast_moe_version` and `GLM53_EXL3_MOE_FAST=0`; passed with distinct gate/up pointers (`thin-off-missing-symbol-proof.json`).

Thus a no-build adaptation is supported for the default profile. Required follow-up is a clean disposable-image composition/static validation of all latest Python overlays and then separately authorized runtime acceptance, not automatically a CUDA rebuild. If operator enables FAST=1, do not silently fall back: require a compiled `glm53_fast_moe_version()==1` image and real GPU gate.

### Build pins if a full rebuild is chosen

- Latest Dockerfile base: `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`.
- EXLLAMAV3 source: `c5d9c657966ffeeaa9353f0cc899f18629da4a13`.
- Compile architecture: `TORCH_CUDA_ARCH_LIST=12.1a`; aarch64 patch → fat-kernel patch → decode-pipeline patch → build.
- Existing required E3 symbols: `exl3_moe`, `exl3_fat_gemm`, `exl3_fat_gemm_scatter`, `exl3_fat_moe_gateup`, `exl3_fat_moe_down`, `exl3_fat_moe_gather`; fused setup also calls `exl3_moe_max_concurrency`.
- Latest Dockerfile installs `instanttensor==0.2.0` with `--no-deps` to avoid replacing NCCL 2.30.7 with package dependency 2.29.7. Version is pinned upstream, wheel hash is not.
- Independently resolved CPython-3.12 arm64 wheel: `instanttensor-0.2.0-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl`; PyPI SHA256 `eefde9b121cd3a2699bd079a19cba5c0a74f23cc1fa2dd8e186a12910f89408e`. URL and complete PyPI metadata saved in `pypi-instanttensor-0.2.0.json`. This is registry metadata verification, not a downloaded-wheel byte check.
- Cooperative TP2 `cooperative_moe.so` and TP3 ABI2 bundles are separate generated/build artifacts; not default requirements.

## 4. Model provenance: do not follow moving draft main

### Target

`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`:

- Upstream old **and latest** launcher pin: `25a44fdbf16862a46b7cc9921142c6c81350af2f`.
- Existing local recipe pin: `024db9f7e9871e8efdf21538ba55af7442be3cd5`.
- Live HF main: `9eaebb7c4e96d983dcd538e18624622ba5b820a8`.
- API per-file `(blobId,size,LFS identity)` comparison: only `README.md` differs between upstream pin and local pin, and only README differs between local pin and live main. Serving weights, config, tokenizer and processor bytes are identical by Hub object identity.
- Keeping the existing immutable local pin is serving-payload faithful and avoids an unnecessary download; document that it is a payload-equivalent mirror revision, not the literal launcher pin. Pinning latest HF main would be a documentation-only update, not required for source fidelity.

### Draft

`incoai/GLM-5.3-Flash-DFlash2`:

- Upstream old **and latest** receipt-matched pin, also local recipe: `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`.
- Live HF main: `bf582e4eacc1810f76656d1811693ff6c6737d2a`.
- Draft main changes weights, not only README: pinned `model.safetensors` LFS SHA256 `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`; main SHA256 `b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac`. Same size 2,342,169,800 bytes does not imply identity.
- **Keep dc77ff1c99eeb2df044ee3d4f0094eb033fee410.** Following main would violate latest launcher's explicit measured-checkpoint choice. Also changes README and adds an ancillary figure.

All API file maps and comparisons are saved in `hf-*.json` / `model-payload-comparison.json`; weights were not downloaded. Keep revision consistent in top-level model declaration, distribution entries and exact target/draft snapshot paths. Local served-model alias remains stable.

## 5. Intentional orchestration / security differences

These need documentation and relevant gates, not blind shell copying:

- SparkRun owns image/model distribution, local immutable cache snapshots, host selection, rank generation, container lifetime and logs instead of upstream's Docker/SSH/rsync shell.
- Native `mp` / two-node topology is the same, not a Ray substitution.
- Port 8000 vs 8888, persistent cache root translation, entrypoint clearing and wrapper supervision are orchestration changes.
- Per-host CX7/HCA/GID detection replaces fixed upstream interface names. Preserve shared NCCL settings and validate each selected GID. New NCCL channel override is optional; do not force 8 without an explicitly selected tuning profile.
- Local image-only `{"image":4,"video":0}` plus `media.invalid` domain allowlist is a stricter security/resource policy than upstream. Preserve it while adding new per-image 2048-token and 1-GiB processor-cache controls.
- Local fail-closed post-ready gate is stronger than upstream's nonfatal boot warmup. Keep its explicit process ownership / termination semantics and update warmup support.
- Upstream memory preflight requires `MemAvailable >= MemTotal * GPU_MEM_UTIL + 2097152 KiB` on both nodes; this is distinct from vLLM's CUDA-free admission gate. Translate as a launch preflight after old workload removal, not a reason to disrupt current hosts during this audit.
- New checkout-scoped lifecycle flock serializes start/restart/stop; stop waits up to 30s, does not kill lock holder; health polling is pipefail-safe and requires three consecutive non-running observations. SparkRun owns analogous lifecycle behavior; no need to install this checkout-local shell lock.
- USER fallback for non-login sessions; warnings on inherited environment overriding critical dotenv keys; diagnostic GLM53_EXTRA_ENV validates names/values and refuses launcher-owned settings. Static literal YAML settings eliminate most dotenv mechanics; do not add unrestricted shell/env injection merely for parity.
- Optional NFS weight sharing remains OFF for TP2. Local snapshot distribution is intentional and avoids NFS runtime / server / export side effects.

## 6. Remaining material changes outside the standard TP2 profile

- New TP3 launcher and padded 64→66-head / expert-shard / model-loader/parameter/backend overlays; NFS support, multi-rank bundle preflight, optional ABI2 cooperative EP and FlashKDA integration. Not applicable to a fixed two-node recipe.
- TP3 KDA FP8 Marlin alignment workaround keeps f_b/g_b projections BF16; ABLIT donor padding mirrors TP3 end-padding. Default TP2 with ABLIT=0 is unaffected.
- TP3/TP4 receive fair scheduling and DFlash SWA-retention support; TP4 optional sparse MLA final-call slice (`VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64`, default 0) mitigates stalls but does not fix underlying race. TP3/TP4 wrappers explicitly avoid inheriting TP2's 14-GiB cap while preserving caller EXTRA_ARGS.
- Optional TP2 cooperative decode adapter requires generated pinned overlay plus runtime.py / cooperative_moe.so staged on every rank; standard E3 prefill remains. Do not select it for default refresh.
- Optional KDA BF16 large-M path requires KDA dense-FP8 selection, BF16/SM121 and qualified TP-local shape; threshold strictly M>512, extra retained memory approximately 3.26 GiB/rank on TP2. Not a new standalone mandatory CUDA extension.
- Optional abliterated preset (`start-abliterated.sh`) pins `bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit` at `14858211ed81d7fa773f8a0db02f38f36d230252`, 120 shards; forbids applying runtime ABLIT again. Different model, not part of requested refresh.
- New kernel lab is development-only (production additions explicitly dropped); tool-concurrency benchmark, concurrency ladder, numerical calibrated-panel tooling and hardened evidence handling, sparse/compact/kernel tests, diagnostic spark_doctor and spec-accept-gate scripts are development/qualification additions rather than serve defaults.
- Documentation grows to support 2–4 nodes and records measured TP2/TP3 prose profiles, compact-mode limitations, numerical uncertainty and a GB10 UVM runbook. Historical benchmark claims are not new-runtime acceptance evidence.
- Latest HEAD includes `examples/tp2-long-coding.env`: **optional, never automatically sourced**. It selects context 262144, sequences 2, batched tokens 1024, utilization 0.865, no 14-GiB cap, FAST=1, KDA_BF16_LARGE_M=1, DENSE_FP8=dense,kda, compact draft=1, spinwait=16, global/SWA retention 14336/0. The retention pair is explicitly unqualified. Do not confuse this example with latest default 850K profile or enable it implicitly.

## 7. Audit limitations / release gates

- Confirmed source delta, immutable remote identities, file payload identity and CPU control-path contracts; not launch success, numerical GPU parity, long-context capacity or performance.
- Public image metadata does not prove installed Python-overlay state after adaptation. Run disposable-image composition/static tests on exact pinned image before launch, retaining clean-base/second-application checks.
- The changed fair scheduler, Mamba alignment/lifetime and cache composition deserve mandatory regression coverage even though FAST/compact remain OFF.
- Do not require the optional thin symbol unconditionally in default-profile preflight; that would create an unnecessary build blocker. Do fail closed if FAST=1 is selected without it.
- Preserve current production workload; do not run full GPU qualification until explicitly authorized. Keep old-profile rollback pins.
- Before final publication, recheck upstream HEAD, recipe/mod checksums, rendered flags/env on both ranks, and the exact final worktree. This audit does not approve edits made after its baseline read.
