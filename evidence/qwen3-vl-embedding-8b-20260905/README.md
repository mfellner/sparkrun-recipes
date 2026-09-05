# Qwen3-VL-Embedding-8B acceptance — 2026-09-05

This directory preserves the bounded live acceptance for `recipes/qwen3-vl-embedding-8b-dgx-spark.yaml` on `gx10`, the otherwise-idle third node in the same physical Spark ring where Qwen3.8 Flash Next occupies `dgx03` and `dgx04`.

## Deployed identity

- SparkRun ID: `sparkrun_7d7b52c2c9174082_15b33f409193`
- Direct API: `http://192.168.178.51:8000/v1`
- Served model: `qwen3-vl-embedding-8b`
- Model revision: `2c4565515e0f265c6511776e7193b22c0968ddc7`
- Image manifest: `sha256:f92b4a1a476fd1e235e97df2a623c04c24b69d607a78a91f5084627ec7bd4266`
- vLLM: `0.28.1rc1.dev441+g2902ca17e.d20260905`
- Configured context: 32,768 tokens
- Output dimension on this runtime: 4,096

## Files

- `acceptance.py`: generates four deterministic PNG fixtures and tests the direct embeddings API.
- `acceptance-result.json`: measured text batching, vector dimensions/norms, four-request concurrency, visual cross-modal retrieval, controlled OCR retrieval, and long-context retrieval results.
- `acceptance-raw.json.gz`: lossless canonical requests and full API responses, including every 4,096-dimensional vector and nanosecond request interval used to recompute the published metrics and concurrency overlap.
- `red-circle-EMBER-742.png` and `blue-square-OCEAN-319.png`: visually distinct multimodal fixtures.
- `ocr-card-EMBER-742.png` and `ocr-card-OCEAN-319.png`: cards with identical layout, shape, and colors; only the rendered code differs, isolating OCR-sensitive ranking.
- `non-regression.py`: checks direct and LiteLLM-proxied completions for the pre-existing GLM-5.3 Flash and Qwen3.8 services.
- `non-regression-result.json`: measured non-regression responses.
- `runtime-audit.json`: machine-readable identity, effective executor/security surface, and log/kernel summaries derived from the raw runtime receipt.
- `capture-runtime.py` and `runtime-receipt.json`: exact post-acceptance capture commands plus raw Docker/image inspection, package versions, model-manifest check, cache identity, process/GPU/memory state, full serve log, kernel scan, and HTTP responses, bound to the acceptance and non-regression run IDs and result hashes.
- `refresh-derived.py`: deterministically rebuilds `runtime-audit.json` and `artifact-manifest.json` from the raw receipts and current artifacts.
- `verify-evidence.py` and `verification-result.json`: recompute every acceptance metric from the compressed raw requests/responses, verify four-way request overlap, and exercise adversarial substitution/deletion/security/timestamp controls.
- `artifact-manifest.json`: required artifact set, byte sizes, and SHA-256 digests consumed by the verifier.

## Acceptance boundary

Passed:

- `/health` and `/v1/models`
- Four-item text batch with 4,096-dimensional, unit-normalized vectors
- Four text embedding requests whose captured nanosecond intervals overlap
- Inline-data image embedding and OCR-sensitive cross-modal ranking
- Controlled OCR ranking where paired image fixtures differ only in rendered code
- A 26,442-prompt-token positive/negative needle-retrieval comparison
- Complete four-shard pinned snapshot with no missing or broken links
- Final-process log scan: zero tracebacks, error-level lines, permission failures, CUDA fatal matches, or NCCL fatal matches
- Boot-kernel scan: zero `NVRM`, `Xid`, or `oom-kill` matches
- Existing GLM-5.3 Flash and Qwen3.8 direct and proxy completions

Advertised/configured but not empirically exercised here: video or mixed-modal input, screenshot-specific behavior, and every supported language.

The checkpoint advertises Matryoshka dimensions from 64 to 4,096, but the pinned vLLM online server returned HTTP 400 for `dimensions: 1024`: `Model 'qwen3-vl-embedding-8b' does not support Matryoshka embeddings`. Use full 4,096-dimensional output through this recipe.

The container image has no Docker healthcheck; acceptance uses the live HTTP health endpoint, model discovery, real requests, process state, and log/kernel scans instead.

## Rerun

```bash
python3 evidence/qwen3-vl-embedding-8b-20260905/acceptance.py \
  --output evidence/qwen3-vl-embedding-8b-20260905/acceptance-result.json

python3 evidence/qwen3-vl-embedding-8b-20260905/non-regression.py

python3 evidence/qwen3-vl-embedding-8b-20260905/capture-runtime.py \
  --output evidence/qwen3-vl-embedding-8b-20260905/runtime-receipt.json

python3 evidence/qwen3-vl-embedding-8b-20260905/refresh-derived.py

python3 evidence/qwen3-vl-embedding-8b-20260905/verify-evidence.py
```

The API is unauthenticated and host-networked, and the recipe enables `--trust-remote-code`. The model revision is immutable, but that pinned code remains part of the trusted supply chain and must be audited before a revision change. The effective SparkRun 0.3.6 container was captured as non-privileged, UID/GID `1002:1002`, `no-new-privileges`, host networking/IPC, `CAP_SYS_PTRACE`, 32 GiB shared memory, all-GPU access, `/dev/infiniband` `rwm` access, and host cache bind mounts; these resolved executor properties must be rechecked after SparkRun or executor changes. Remote URL media is blocked by the reserved `.invalid` allowlist; inline `data:` media is accepted. Keep the endpoint on the trusted private LAN.
