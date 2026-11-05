# Cooperative MoE: benchmark and validation report

Operator note for the live 2026-09-16 serve (geometry 1, rollback, pins):
[cooperative-moe-handoff.md](cooperative-moe-handoff.md). To reproduce the
setup on an existing installation, follow the
[two-node opt-in and rollback guide](cooperative-moe-quickstart.md). Activation
is a selected overlay (`EXL3_OVERLAY_HOST`), not a standalone environment
toggle.

## Configuration and provenance

This is a GLM-5.3-Flash specialization of Turboderp's two-stage cooperative MoE
kernel ([exllamav3 `58d4d732`](https://github.com/turboderp-org/exllamav3/commit/58d4d7322a1b3bd70aae8412487b21cc5e205cf4),
MIT). Headers are archived from `02aef45cd681b960a00afcd0749a4ab99e6c1bfe`. The
DS4.1 2× Spark extension
([MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks#8](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/pull/8))
is the integration model. Its `.so` is compile-time locked to DeepSeek-V4.1 and
is not used here.

Evidence for the GLM constants (not guesswork):

| Field | Value | Source |
|---|---|---|
| Hidden | 4096 | `overlay/exl3.py` `_exl3_hidden_size`; `ablit/LAYER_MAP.json` |
| Packed intermediate | 2048 | README routed `moe_intermediate_size`; TP2 column/row shard |
| Local intermediate | **1024** | `create_weights(..., intermediate_size_per_partition)`; `tests/bench_e3_microbench.py` `INTER=1024` |
| Top-k | 8 | `EXL3_FAT_GROUPED_TOPK` default; same microbench `TOPK=8` |
| Quant | K4 MCG, no mul1 | overlay codebook=mcg; kernel cb=1 vs mul1 cb=2 (`exl3_moe_coop_prepare`, `codebook.cuh`) |
| Decode rows | 1–32 | DFlash2 k=7 → 8 rows/seq; `MAX_NUM_SEQS=4` → 32; `EXL3_TEMP_ROWS_FUSED=32` with E3 |
| Prefill | stock E3 | `apply_exl3_fused_moe` only uses grouped fat when `tokens > cap`; coop wraps decode-sized fused only |
| Geometry | **A wide, B wide (1)** | DS4.1 serve used geometry 1. Blackwell auto is A-wide/B-narrow (`I/16=64`) and launches 4× down blocks on H=4096; that is a decode loss on this kit |

Thin vs fat in `apply_exl3_fused_moe`: `tokens <= cap` is one fused `exl3_moe`
launch (the cooperative target). `tokens > cap` with `EXL3_FAT_GROUPED=1` keeps
thin experts on fused `exl3_moe` and every fat expert on E3. Cooperative
`call_eligible` requires 1–32 rows and top-k 8, so prefill never takes this
path.

Native ABI: `glm53_coop_abi` / `glm53_coop_info` / `glm53_coop_launch` (v1).
The historical measured library had digest `aa3fe5e9…` (nvcc 13.0 in image
`sha256:eecb36e…`). Current builds are accepted by `build-manifest.json` source
provenance rather than that complete ELF hash: pinned ExLlamaV3 commit,
combined tree, repository source hashes, SM121a target and compiler identity.
Runtime independently verifies ABI, layout and occupancy.

## Stock decode baselines (this kit)

sparkDash, DFlash2 k=7, fused EXL3 MoE. These are the numbers to beat; they are
**not** cooperative measurements.

| Workload | Concurrency | Stream tok/s | Aggregate tok/s | When |
|---|---|---:|---:|---|
| Structured (count 1→200) | ×1 | 62.9 | 62.9 | 2026-08-28 |
| Prose (adaptive-k + dense FP8) | ×1 | 32.1 | 32.1 | 2026-09-08 |
| Prose (adaptive-k + dense FP8) | ×2 | 22.1 | 41.2 | 2026-09-08 |

Protocol: temp 0, thinking off, 400 tokens, CUDA graphs. Prose used
`GLM53_ADAPTIVE_K=ema`, `GLM53_DENSE_FP8=dense,kda`, 850k context, 14 GiB KV
cap. Structured used the 2026-08-28 1M-pool serve.

## Cooperative serving measurements

Lab `tests/bench_decode.py` on this two-node serve (adaptive-k + dense FP8,
DFlash2 k=7, E3 on, CUDA graphs). Two warmed runs. Geometry 1 is the adapter
default.

Blackwell auto (geometry 2, A-wide/B-narrow) was a decode loss here: the down
grid is `rows*topk*(H/32)` — 8192 blocks at 8 rows — versus 2048 with wide B.
Stock fused EXL3 is already near this kit's copy ceiling, and DFlash2 ×1 into
288 local experts has almost no weight reuse, so the extra B blocks read as
“no benefit” on prose.

| Workload | Stock | Coop geo 2 (auto) | Coop geo 1 (wide/wide) |
|---|---:|---:|---:|
| Structured (count 1→200, 200 tok) | 62.9 sparkDash | 78.0 | **80.0** |
| Prose (hash-map, 400 tok) | 32.1 (2026-09-08) | 34.8 | **37.4** (+16% vs 32.1) |

GPU util during decode stays ~96% (kernel-bound). Prefill is unchanged (E3).
sparkDash prose ×2 is still open.

## Numerical and safety validation

Historical host-side qualification had 49 dispatch assertions and eight
profile-integrity tests. Current coverage adds source-manifest drift,
architecture/symbol checks and acceptance of different valid compiler output;
`bash -n` covers `build.sh`.

Native rebuild: K4 / cb=1 (`mcg`) kernels for wide and narrow tiles plus the
rotation kernel, `sm_121a`. Occupancy is checked at `glm53_coop_info` before
scratch is published.

GPU integration (this head, 2026-09-16, image `sha256:9581c4c7…`, maintenance
container, no checkpoint load): **18/18** cases passed the 0.3%-of-reference-peak
screen. Cooperative was selected for rows 1, 2, 4, 7, 8, 16, 24, 32 (spread and
concentrated routing, including CUDA-graph replay/mutation and invalid routes).
Rows 40 stayed stock (E3 grouped remains the prefill/fat path). Load-time E3
resolution stayed `effective_tier=grouped`. Strict differences vs stock fused
were retained (6,473,169 raw / 4,140,219 post-BF16 failed elements across
repeated comparisons) — not bit-exact, as documented. A two-node serving smoke
test has been executed (geometry 1 on both ranks).

Current source-build qualification (2026-09-28, image `sha256:0c2bcbe9…`): two
isolated builds were byte-identical at `03bb1293…`. That artifact passed the
48-check numerical/CUDA-graph gate separately on both Sparks, including all
live capture sizes and grouped fallback at rows 40. The bounded 8-check
`compute-sanitizer --tool memcheck` gate also passed on both GPUs with zero
errors. These isolated rank tests do not claim distributed serving or quality
validation.

A subsequent matched TP2 comparison held the tuned serving configuration fixed
and changed only the cooperative overlay. Against `GLM53_EXL3_MOE_FAST=1`, the
candidate improved structured decode by 2.0% at ×1 and 1.8% aggregate at ×2.
Hash-map prose was 3.1% slower at ×1 and tied at ×2; the warm 32,753-token Pi
replay was 6.0% slower because small numerical differences reduced DFlash2
acceptance. Direct CUDA-event tests still showed the intended kernel shape:
1.32× at one spread row, 1.04× at eight, parity at 32, and 1.08–1.33× with
expert reuse. The extension is therefore a valid opt-in, but FAST remains the
production choice on this workload.

## Remaining work

- Bounded racecheck for the current source-built artifact.
- Real-weight arithmetic and assistant-output quality vs stock.
- Prolonged production burn-in. Default serving remains stock until the overlay
  is selected.
