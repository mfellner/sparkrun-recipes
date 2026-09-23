# Did v5 fair mixed-prefill fix the reporter hang?

**Independent retest (TP2):** 2026-09-15 ~12:26–12:36 local.  
**TP3 follow-up:** 2026-09-15 ~14:26–14:35 local, same v5 overlay, live `glm53-exl3-tp3-head`.  
**Serve already on v5** (`glm53-exl3-head`, EngineCore pid 2019); no extra restart. `/health` 200.  
**Overlay:** `# [glm53-decode-floor:v5]`, `GLM53_MIXED_PREFILL_CHUNK=fair`, chunk 256, share 0.20, interval 2000 ms, max step 1000 ms, max chunks 1, `LONG_PREFILL_TOKEN_THRESHOLD=3584`.  
**Harness:** `logs/fair-v5-20260915/overlap_arms.py`  
**Receipts:** `logs/fair-v5-retest-20260915/` and `/tmp/mixed-prefill-fair-v5-retest.json`  
**Prior pass (other agent):** `logs/fair-v5-20260915/` — numbers agree in direction; this file’s tables are the independent rerun.  
**Design:** [astra-fix.md](astra-fix.md)

## Verdict

**Yes, for the reporter freeze, when `fair` v5 is live.** A 2k unique newcomer submitted 10 s into a thinking-on essay (`max_tokens=4000`) got its first token in **11.6 s, while the essay was still streaming**. Under `skip` that request waits for the whole essay (~163 s here; 221 s / 20+ min in the original report).

A ~30k unique newcomer on the same recipe also finished **during** the essay (**163.6 s TTFT**, 16 s before A’s last token), with **all 29.4k prompt tokens** in mixed steps. That duration is the 20 % share budget, not Waiting starvation.

Incumbent decode did **not** collapse to the reporter’s 0.8–2.8 tok/s under `CHUNK=0`. On the essay, overlap rate stayed **~20.6–20.9 tok/s** vs ~24.7 solo (about **1.18–1.20×**). In-run, vs the 10 s of decode before B: **1.06×** (2k B) and **1.21×** (30k B).

`start.sh`, `start-tp3.sh`, `start-tp4.sh`, and the matching `.env*.example` files default `GLM53_MIXED_PREFILL_CHUNK` to **`fair`**. TP=3 overlap is measured below. TP=4 is unmeasured and inherits the same default; set `skip` to restore decode-only isolation.

| Claim | `skip` | v4 `fair` (prior) | **v5 `fair` (this retest)** |
|---|---|---|---|
| 2k newcomer 10 s into thinking essay | waits the whole decode | 21.9 s, during A | **11.6 s, during A** |
| ~30k newcomer 10 s into thinking essay | waits the whole decode, then ~23 s solo prefill | 186 s, after A | **163.6 s, during A**; 29 365 mixed tokens, 0 solo |
| `CHUNK=0` drops A ~10–36× | N/A | A held | A held: essay **1.18–1.21×**; unique-800 **1.21–1.34×** whole-window |

## Live confirm

```
patched scheduler.py (# [glm53-decode-floor:v5])
[glm53-decode-floor] fair v5 probe_chunk=256 ladder=128..2048 share=0.2 interval_s=2.0 max_step_s=1.0 max_chunks=1
```

Container env matched the knobs above. GPU was idle (0 %) before the retest.

## What was run

Same arms as the other agent, thinking/temp as specified, **fresh unique-word prefixes** (not cache-cheap `the`). C1 = A solo; C2 = A then B. Reporter B is submitted **10 s after A’s first token**.

| Arm | A | B | B delay |
|---|---|---|---|
| cold2k | unique ~1.9k, thinking off, max 800 | unique ~1.9k, thinking off, max 8 | at A first token |
| cold30k | unique ~29–30k, thinking off, max 800 | unique ~29–30k | at A first token |
| rep2k | thinking-on essay, max 4000 | unique ~1.9k | +10 s |
| rep30k | same A type (C1 reused) | unique ~29.4k | +10 s |

## Independent results

Overlap tok/s is stream-event rate scaled by `(completion_tokens − 1) / (events − 1)`. Gaps are A’s inter-event gaps in the overlap window (until B’s first token, or A’s last if B is later).

| Arm | C1 tok/s | A overlap tok/s (vs C1; vs in-run before B) | A gaps p50 / p95 / max | B TTFT | B during A? | B mixed progress |
|---|---:|---|---|---:|---|---|
| **rep2k** | 24.7 | **20.9** (1.18×; **1.06×** vs 22.2 before; 25.0 after) | 0.09 / 0.19 / **1.39 s** | **11.6 s** | **Yes** (−141 s vs A last) | 4 mixed steps, **1892 / 1892**, chunks 100 / 256 / 768 |
| **rep30k** | 24.7 | **20.6** (1.20×; **1.21×** vs 24.8 before; 24.3 after) | 0.09 / 0.19 / **1.44 s** | **163.6 s** | **Yes** (−16 s vs A last) | 39 mixed steps, **29 365 / 29 365**, chunks 117 / 768 / 832 |
| cold2k | 65.4 | 41.7 until B first (whole-window A **54.1**, 1.21×) | 0.12 / 0.40 / 1.12 s | **8.0 s** | **Yes** | 3 mixed steps, **1883 / 1883**, chunks 91 / 768 / 1024 |
| cold30k | 67.7 | **50.3** (1.34×) | 0.11 / 0.23 / 0.82 s | 37.2 s | **No** (A decode only 15.8 s) | 5 × **768 = 3840** mixed, then 25 600 solo |

Whole-window A on rep2k stayed **24.5 tok/s** (B left after 12 s of a 163 s decode). Whole-window A on rep30k was **21.1 tok/s** over a stretched **190 s** decode because B occupied almost the entire essay.

## Reading

- **The 20-minute / 221 s Waiting lockout is gone** on this v5 process. Scheduler logs during overlap are mixed `completed_step` rows with 768–1024-class chunks, not `skip` zeros.
- **v5 is using the step budget.** This retest’s mixed sizes are 768 / 832 / 1024, not v4’s stall at 128. That is why 2k TTFT dropped from ~16–22 s (v4) to **8–12 s**.
- **30k during an 800-token A still cannot finish in 16 s.** It moved **3840 tokens** in that window (v4 moved ~1k), then finished solo. Same capacity story as [astra-fix.md](astra-fix.md): at 20 % share, 30k needs on the order of **~2–3 minutes** of continuous mixed service. The reporter essay is long enough; a short 800-token decode is not.
- **Max gap ~1.1–1.4 s** is one mixed step against `MAX_STEP_MS=1000`. p95 stays on the normal decode cadence (~0.19 s on the essay).
- **cold2k overlap 41.7 tok/s** is the short window that contains the 1024-token step; after B’s first token A recovered to **70 tok/s**. Do not treat 41.7 as the whole-request decode rate.

## vs the other agent’s first v5 pass

Same process, new unique prefixes. Direction matches. This retest’s reporter 2k was **11.6 s** vs their **9.0 s**; reporter 30k **163.6 s** vs their **146.4 s**. Chunk ladder and “B during A” are the same. Single-run scatter of that size is expected.

## TP3 live `fair` v5 (2026-09-15 ~14:26–14:35 local)

**Serve:** `glm53-exl3-tp3-head`, EngineCore pid 2286, already on v5; no extra restart. `/health` 200 before and after. `GPU_MEM_UTIL=0.80`, `DFLASH_DRAFT_TP=1`. Same knobs as TP2 (`fair`, chunk 256, share 0.20, interval 2000 ms, max step 1000 ms, max chunks 1).  
**Harness:** `logs/fair-v5-tp3-20260915/overlap_arms.py` (same arms, `HEAD=glm53-exl3-tp3-head`).  
**Receipts:** `logs/fair-v5-tp3-20260915/` and `/tmp/mixed-prefill-fair-v5-tp3.json`. Wall ~8.7 min, all four arms `error: null`, engine still running afterward.

### TP3 verdict

**Same qualitative fix as TP2, and the overlap numbers are better.** A 2k unique newcomer 10 s into a thinking-on essay got its first token in **8.7 s, while the essay was still streaming**. A ~29.5k unique newcomer on that recipe finished **during** the essay (**110.0 s TTFT**, 27 s before A’s last token), with **all 29 479 prompt tokens** in mixed steps.

Incumbent decode did not collapse. Essay overlap stayed **~24.7–25.8 tok/s** vs 29.7 solo (**1.15–1.20×**). Unique-800 overlap stayed **~60.0–63.9 tok/s** vs 77–82 solo (**1.21–1.37×**). Max inter-event gap on every arm was **0.84–0.95 s**, inside the 1.0 s step budget (TP2’s reporter max was 1.39–1.44 s).

The engine did **not** die during these overlap arms. The earlier TP3 “crash on warmup” was a race with `post_ready_warmup` after `/health` 200, not this mixed-prefill recipe.

### TP3 results

| Arm | C1 tok/s | A overlap tok/s (vs C1; vs in-run before B) | A gaps p50 / p95 / max | B TTFT | B during A? | B mixed progress |
|---|---:|---|---|---:|---|---|
| **rep2k** | 29.7 | **24.7** (1.20×; **1.19×** vs 29.4 before; 30.9 after) | 0.075 / 0.150 / **0.93 s** | **8.7 s** | **Yes** (−113 s vs A last) | 4 mixed steps, **1881 / 1881**, chunks 89 / 256 / 768 |
| **rep30k** | 29.7 | **25.8** (1.15×; **1.20×** vs 31.0 before; 31.4 after) | 0.076 / 0.154 / **0.92 s** | **110.0 s** | **Yes** (−27 s vs A last) | 31 mixed steps, **29 479 / 29 479**, chunks 103 / 192 / 768 / 1024 |
| cold2k | 81.8 | **60.0** (1.37×); after B first, A recovered to 84.2 | 0.093 / 0.183 / 0.95 s | **6.5 s** | **Yes** (−4.9 s vs A last) | 3 mixed steps, **1870 / 1870**, chunks 78 / 768 / 1024 |
| cold30k | 77.1 | **63.9** (1.21×) | 0.092 / 0.183 / 0.84 s | 34.0 s | **No** (A decode only 12.5 s) | 4 × **1024 = 4096** mixed, then 25 478 solo |

Whole-window A on rep2k stayed **30.3 tok/s** (B left after ~9 s of a 132 s decode). Whole-window A on rep30k was **27.2 tok/s** over a stretched **147 s** decode because B occupied most of the essay.

### vs TP2 `fair` v5 (independent retest above)

| Arm | TP2 B TTFT | **TP3 B TTFT** | TP2 during A? | TP3 during A? | TP2 A overlap | TP3 A overlap |
|---|---:|---:|---|---|---|---|
| rep2k | 11.6 s | **8.7 s** | Yes | Yes | 20.9 (1.18×) | 24.7 (1.20×) |
| rep30k | 163.6 s | **110.0 s** | Yes | Yes | 20.6 (1.20×) | 25.8 (1.15×) |
| cold2k | 8.0 s | **6.5 s** | Yes | Yes | 41.7 until B first | 60.0 |
| cold30k | 37.2 s | **34.0 s** | No | No | 50.3 (1.34×) | 63.9 (1.21×) |

TP3 mixed prefill on the reporter 30k is about **268 tok/s** effective (29 479 / 110 s) vs TP2’s **~179 tok/s**. Chunk ladder still climbs to 768–1024; first mixed chunk is still clipped (78–103 tokens). Short unique-800 A still cannot absorb a 30k newcomer: TP3 moved **4096** mixed tokens in that window (TP2 moved 3840), then finished solo. Same 20 % share capacity story as [astra-fix.md](astra-fix.md).

GitHub launchers now default TP=3 and TP=4 to `fair` as well. The overlap recipe is healthy on an already-ready TP=3. Boot/warmup races (hitting the API at `/health` 200 before `post_ready_warmup` finishes) are a separate issue; wait for the launcher ready line.

## Still open

- Single runs only (no three-repeat matrix, no multi-newcomer).
- TP4 `skip` / `fair` not measured.
- First mixed chunk is sometimes clipped (78–117 tokens); later rungs recover.
- Host busy-time proxy; TP2 had one mixed step ~1.4 s against a 1.0 s budget. TP3 max accounted mixed step in this pass was 0.95 s.

PR [186](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/186) at v2 contains neither v4 nor v5.
