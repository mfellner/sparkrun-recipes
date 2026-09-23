# Experiment ledger — GLM cooperative MoE decode

Protocol: change one hypothesis at a time. Retain winners, revert losers.
Do not treat geometry 1 as guaranteed fastest. Investigation-complete is
distinct from performance-improved.

| ID | When | Hypothesis | Change | Result | Keep? | Evidence |
|---|---|---|---|---|---|---|
| C1-init | 2026-09-16 | Required native init must not live in `assert` | Move ABI/info/SHA/capability/layout checks to explicit exceptions; add `python -O` regression | Host tests pass; live serve still on preserved geo1 artifacts until the next validated deploy | yes | `extensions/cooperative_moe/test_dispatch.py`, `test_optimized_init.py` |
| C1-scratch | 2026-09-16 | Shared scratch must not overlap | Document contract; reject ubatch/dbo env and EXTRA_ARGS at install | Host tests pass | yes | `runtime.py` `enforce_serialized_execution` |
| C1-log | 2026-09-16 | Installation log is not execution proof | Prepared-layer summary, geometry tiles, capture-size selection (explicitly not replay counts) | Host tests pass | yes | `test_dispatch.py` capture/eager diagnostics |
| C1-geo-env | 2026-09-16 | Geometry must be selected before prepare/capture and forwarded to both ranks | `GLM53_COOP_GEOMETRY` parsed in adapter; launcher validates and forwards | Host tests pass | yes | `start.sh`, `.env.example` |
| C1-native-assert | 2026-09-16 | Slot ceiling should fail at compile time | `static_assert` on 256-slot / 8-row run builder / 344-byte params / 3587 counters | Source only until the next native rebuild | pending rebuild | `cooperative_moe.cu`, `cooperative_moe_kernel.cuh` |
| C1-gpu-gate | 2026-09-16 | Geometry-1 vs stock on every live capture size including 3 and 5 | Full `test_cuda_integration.py` on both Sparks | pass, 48 checks; rows 33/40 concentrated execute E3 grouped; graph replay matches eager | yes | `logs/astra-goal-20260916/receipts/cuda-{head,worker}-pass.log` |
| C1-sanitizer | 2026-09-16 | Bounded memcheck/racecheck of K4/MCG 256-slot kernel | SANITIZER=1 suite: rows 1/8/32/40, concentrated, invalid/zero routes | memcheck 0 errors; racecheck 0 hazards (shared-memory only) | yes | `logs/astra-goal-20260916/receipts/{memcheck,racecheck}-{head,worker}.san` |
| C1-loop-ref | 2026-09-16 | Independent LinearEXL3 loop is a denser oracle than fused stock | Compare coop vs `fused=False` at limit=10 for rows 1 and 8 | peak≈0.001 / rel_l2≈0.001; clip/zero-limit loop diverges (recorded, not a gate) | yes | `logs/astra-goal-20260916/receipts/cuda-{head,worker}-loop.log` |

| E1 | 2026-09-16 | Geometry 0/1/2 kernel times differ at live decode shapes | Isolated CUDA-event microbench, same tensors, geos 0/1/2 at rows 1/3/5/8/32 | Geo1 fastest at 32 rows vs stock (median 0.997 vs 1.183 ms). Geo0 slower than stock at 32 rows. Rows 3 and 5: geos 0/1/2 tied (~0.517 / ~0.77 ms coop). No later geometry beat geo1 | keep geo1 | `logs/astra-goal-20260916/receipts/microbench-head.jsonl` |
| E2 | 2026-09-16 | Routing reuse / concentrated experts would beat spread geo1 | Same microbench, concentrated vs spread, 32 and 288 experts | Concentrated is faster for both stock and coop; geo1 still wins vs stock. Does not justify a new kernel | no extra kernel | same jsonl (concentrated / 288-expert rows) |
| E3 | 2026-09-16 | Adaptive-k sizes 3 and 5 need a different tile | Microbench rows 3 and 5 across geos 0/1/2 | Tied across geometries; serving already captures 3 and 5 on geo1 | keep geo1 | same jsonl |
| C1-serve | 2026-09-16 | C1 (geo1 `.so` + runtime fixes) beats matched stock serving without shrinking frozen knobs | A/B/A stock ↔ C1, 5 warmed runs, prose/structured/coding ×1 and ×2 | C1 faster on matched structured/prose/coding; independent boot2 and 2026-09-16 16:21 C1-final revalidation agree. Quality pass. Mixed 8k prefill + 12 sequential HTTP 200. KV still 883552 tokens / 14 GiB. E3 grouped still configured; oversized rows deselected from coop | **yes (winner)** | `receipts/bench-{stock,c1,c1-boot2,c1-final}/`, `quality-c1-final.json`, `mixed-sustained-c1-final.json` |
| E-later | 2026-09-16 | A newer cooperative kernel would beat the initial geo1 `.so` | Further geometry/kernel investigation on this kit | **Investigation complete and negative.** Serving winner remains C1 = initial geo1 binary `aa3fe5e9…` plus C1 runtime. Distinguish: improved vs stock; not improved vs the first geo1 kernel | no | microbench + serving A/B/A |

Kernel binary throughout is the preserved geo1 `.so` (`aa3fe5e9…`). C1 changes the adapter (assert-init, scratch contract, capture-size logging, `GLM53_COOP_GEOMETRY`), not the native library.

Snapshot / rollback: `logs/astra-goal-20260916/` (`rollback-stock.sh`, `rollback-geo1.sh`, `rollback-c1.sh`).
Acceptance criteria: [astra-acceptance.md](astra-acceptance.md) (frozen gates; not rewritten for this result).
