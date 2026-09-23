# GLM cooperative MoE decode results (2026-09-16)

Winner: **C1** = geometry-1 cooperative kernel (same `.so` as the first geo1
deploy) plus runtime fixes (initialization outside `assert`, scratch
non-overlap contract, prepared-layer and capture-size logs,
`GLM53_COOP_GEOMETRY`).

Performance **improved versus matched stock** on this two-Spark TP2 serve.
Further MoE geometry/kernel work **did not beat** that initial geo1 binary.
Investigation of extra kernel wins is complete and negative.

DeepSeek was not modified. Frozen serving knobs were not shrunk
(checkpoint, EXL3 K4 MCG, TP2, E3 grouped prefill, DFlash2 k=7 adaptive-k
`2,4,7`, CUDA graphs, `MAX_MODEL_LEN=850000`, KV 14 GiB / 883,552 tokens,
`MAX_NUM_SEQS=4`). Gates: [astra-acceptance.md](astra-acceptance.md).
Ledger: [astra-experiment-ledger.md](astra-experiment-ledger.md).

## Reproduction

```bash
# Health and identity
curl -fsS --max-time 8 http://127.0.0.1:8888/health
curl -fsS http://127.0.0.1:8888/v1/models   # GLM-5.3-Flash-EXL3
sha256sum ~/.cache/vllm-glm53-flash/cooperative_moe/{cooperative_moe.so,runtime.py,exl3-cooperative.py}
ssh -o BatchMode=yes zurih@10.0.0.2 \
  'sha256sum ~/.cache/vllm-glm53-flash/cooperative_moe/{cooperative_moe.so,runtime.py,exl3-cooperative.py}'

# Quality (temp 0, thinking off, assistant text)
python3 tests/eval_coop_quality.py \
  --out logs/astra-goal-20260916/receipts/quality-c1-final.json --label c1-final

# Decode benches (matched protocol: warmup, 5 runs, skip-coherence on structured/×2)
python3 tests/bench_decode.py --phase structured --structured --runs 5 --max-tokens 400 \
  --skip-coherence --out logs/astra-goal-20260916/receipts/bench-c1-final/structured-x1.json
python3 tests/bench_decode.py --phase structured-x2 --structured --concurrency 2 --runs 5 \
  --max-tokens 400 --skip-coherence --out logs/astra-goal-20260916/receipts/bench-c1-final/structured-x2.json
python3 tests/bench_decode.py --phase prose --runs 5 --max-tokens 400 --skip-coherence \
  --out logs/astra-goal-20260916/receipts/bench-c1-final/prose-x1.json
python3 tests/bench_decode.py --phase prose-x2 --concurrency 2 --runs 5 --max-tokens 400 \
  --skip-coherence --out logs/astra-goal-20260916/receipts/bench-c1-final/prose-x2.json

# Restore C1 after a stock experiment (GLM only)
logs/astra-goal-20260916/rollback-c1.sh
```

Image: `glm53-flash-sm121:e3-20260907-itensor`. API
`http://127.0.0.1:8888`. Auth none on this kit.

## Provenance (live C1, 2026-09-16 final boot)

| Artifact | SHA-256 |
|---|---|
| `cooperative_moe.so` (head and worker) | `aa3fe5e9387c7e0d42d685fb2ca8a5fb959ad956600236baac078a9076c17a1c` |
| C1 `runtime.py` (head and worker) | `9427f6a65def09ebdbea231e42f735236e145f3d02c19cf5e5276c2e704ce1ca` |
| Generated overlay `exl3-cooperative.py` | `5f28f5543629043117c7506cd3cf47cc8fa66fcd1153ddf67c696bf3e5484c4d` |
| Stock `overlay/exl3.py` (unchanged recipe copy) | `fe07cf3cd1928d0a189e793579a7d2dd529f75617a55f620ee14a0a9d3b20121` |

The geo1 kernel hash matches the sanitizer/GPU-gate binary; sanitizers were
not rerun after that hash-stable redeploy. Adapter digest `9427f6a6…` is the
C1 runtime (original geo1 adapter was `71111c23…`). Goal-start snapshot:
`logs/astra-goal-20260916/provenance.json`.

Both ranks, final boot (`receipts/start-c1-final.log` / docker logs):
`cooperative MoE prepared-layer summary: eligible=42 … geometry=1 (A-wide/B-wide)`,
`GPU KV cache size: 883,552 tokens`, `fat_grouped=1`, capture sizes
`1,2,3,4,5,6,8,9,10,12,15,16,20,24,32`.

## Serving A/B/A (median of 5 warmed runs)

`decode_ms_per_draft_step` is a serving-cycle ratio
(`1000 * decode_s / spec.drafts`), not a CUDA kernel timer. Min/max show
run-to-run spread. Independent C1 boot2 and C1-final ×2 revalidation are
listed separately.

### Structured (count 1–200, 400 completion tokens, mean_draft_tokens_per_step = 7.0)

| Arm | conc | tok/s median (min–max) | TTFT s | decode ms/step | accept/step | accept ratio | agg tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock | 1 | 72.03 (70.63–72.92) | 0.348 | 108.42 | 6.863 | 0.980 | — |
| C1 boot1 | 1 | 77.18 (76.16–77.47) | 0.304 | 99.18 | 6.712 | 0.959 | — |
| C1 boot2 | 1 | 77.29 (75.66–79.01) | 0.310 | 99.28 | 6.712 | 0.959 | — |
| stock | 2 | 60.18 (59.73–61.95) | 0.327 | 63.98 | 6.796 | 0.971 | 113.91 |
| C1 boot1 | 2 | 65.90 (64.58–67.17) | 0.323 | 58.67 | 6.777 | 0.968 | 124.38 |
| C1 boot2 | 2 | 66.00 (64.25–67.80) | 0.330 | 58.13 | 6.786 | 0.970 | 125.47 |
| C1 final | 2 | 65.91 (65.55–67.53) | 0.307 | 58.78 | 6.786 | 0.970 | 124.46 |

Structured ×1 is about **+7% tok/s** vs stock with **lower** ms/step at
nearly the same draft length (k=7). That is the cleanest serving comparison.

### Prose (hash-map, 400 completion tokens)

| Arm | conc | tok/s median (min–max) | TTFT s | decode ms/step | accept/step | mean draft tok/step | agg tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock | 1 | 34.09 (31.35–34.94) | 0.316 | 96.72 | 2.314 | 4.182 | — |
| C1 boot1 | 1 | 36.18 (36.08–37.28) | 0.304 | 87.03 | 2.181 | 4.153 | — |
| C1 boot2 | 1 | 35.25 (34.28–35.86) | 0.230 | 85.57 | 2.023 | 3.926 | — |
| stock | 2 | 25.00 (23.93–25.24) | 0.354 | 61.03 | 2.073 | 3.744 | 48.49 |
| C1 boot1 | 2 | 26.66 (26.36–27.43) | 0.412 | 56.50 | 2.038 | 3.691 | 51.46 |
| C1 boot2 | 2 | 27.00 (26.73–29.06) | 0.339 | 55.47 | 2.077 | 3.762 | 52.38 |
| C1 final | 2 | 27.16 (25.70–29.24) | 0.407 | 56.29 | 2.105 | 3.812 | 52.61 |

Prose throughput is higher on C1; acceptance also moves slightly. Report
acceptance separately; do not treat tok/s as a pure kernel delta.

### Coding (clamp_range; completion length is not matched — stop earlier on some arms)

| Arm | conc | tok/s median | TTFT s | decode ms/step | accept/step | agg tok/s | completion tokens median |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock | 1 | 46.86 | 0.363 | 110.93 | 4.238 | — | 185 |
| C1 boot1 | 1 | 51.45 | 0.341 | 97.84 | 4.086 | — | 178 |
| C1 boot2 | 1 | 51.26 | 0.360 | 100.07 | 4.119 | — | 324 |
| stock | 2 | 35.43 | 0.401 | 68.38 | 3.837 | 56.11 | 518 |
| C1 boot2 | 2 | 38.25 | 0.407 | 61.99 | 3.915 | 64.28 | 506 |

Coding ×1 token counts differ across boots; prefer structured/prose for
attribution. Direction matches C1 faster.

Receipts: `logs/astra-goal-20260916/receipts/bench-{stock,c1,c1-boot2,c1-final}/`.

## Kernel microbench (CUDA events, not serving)

Head-node isolated stock vs cooperative, 50 iters. At **32 rows**, geo1 beats
geo0 and geo2. At **3 and 5 rows** the three geometries are tied. Geo1 vs stock
at 32 rows: about 1.18× (0.997 vs 1.183 ms median).

Receipt: `logs/astra-goal-20260916/receipts/microbench-head.jsonl`.

## Quality, mixed prefill, sustained

| Check | Result | Receipt |
|---|---|---|
| Assistant quality C1-final | all probes pass, no NaNs | `quality-c1-final.json` |
| Quality C1 boot2 / stock | pass | `quality-c1-boot2.json`, `quality-stock.json` |
| Mixed ~8k then decode | tokenize 8192, usage prompt 8206, reply `OK`, HTTP 200 | `mixed-sustained-c1-final.json` |
| Prefill path | chunks 3584+3584+960+78; coop `rows_out_of_range`; E3 `effective_tier=grouped` at prepare | docker `glm53-exl3-head` + start log |
| KV after mixed | still 883,552 tokens / 15032385536 bytes | `/metrics` in same receipt |
| 12 sequential chats | HTTP 200, no NaNs, health 200 | same receipt |

GPU gates (not rerun; `.so` hash unchanged): `cuda-{head,worker}-pass.log`,
`-loop.log`, `{memcheck,racecheck}-{head,worker}.san`.

## Rollback

GLM-only. Do not stop DeepSeek.

| Script | Restores |
|---|---|
| `logs/astra-goal-20260916/rollback-c1.sh` | C1 adapter + geo1 `.so` + overlay, then `start.sh` |
| `logs/astra-goal-20260916/rollback-geo1.sh` | original geo1 adapter (`71111c23…`) + same `.so` |
| `logs/astra-goal-20260916/rollback-stock.sh` | stock overlay `.env` |

## Remaining limitations

- Cooperative is not bit-exact with stock EXL3; quality is behavioral plus the
  GPU peak/L2 screens, not real-weight activation dumps on every layer.
- Graph-replay counters cannot count cooperative launches; capture-size
  selection logs prove what was captured.
- `grouped_calls` is not reprinted on every live prefill; oversized-row
  deselection plus `effective_tier=grouped` and 3584-token chunks are the
  live serving evidence that E3 still handles fat prefill.
- Compile-time `static_assert` slot-ceiling is source-only until a native
  rebuild (binary kept hash-stable on purpose).
- Leftover decode time is still attention / draft / NCCL, not unread expert
  weights. Do not expect another large jump from this kernel on K4 fused.
