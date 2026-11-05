# Adopted 2026-09-26 (TP2): `GLM53_DENSE_FP8=all` + KDA BF16 large-M prefill at an 11 GiB KV pool

Dated finding (AGENTS §5.8). Scope: **TP=2 on 2× GB10 only**; TP=3 is not changed (see the end). Labels:
**Measured** / **Code-verified** / **Interpretation**. Receipts are in gitignored run directories under `logs/`.

## What changed

Production `.env` on the TP2 cluster, 3 lines (HEAD `f4970207` unchanged; `.env` `c3c3aba1` → `31296c97`), and the
matching `.env.example` defaults in this commit:

| Setting | Before | After |
|---|---|---|
| `GLM53_DENSE_FP8` | `dense,kda` | `all` |
| `GLM53_KDA_BF16_LARGE_M` | `1` but inert (live cooperative overlay `682ed186` predates #233) | `1`, active |
| `EXL3_OVERLAY_HOST` (cluster-local) | `~/.cache/vllm-glm53-flash/cooperative_moe/exl3-cooperative.py` (`682ed186`) | `~/.cache/vllm-glm53-flash/exl3-cooperative-refreshed-895e5269.py` (repo `overlay/exl3.py` `849e2588` + the 6-line cooperative footer) |
| `--kv-cache-memory-bytes` | `15032385536` (14 GiB) | `11811160064` (11 GiB) |

The cooperative overlay is a generated local artifact and is not in the repo. Regenerate it with
`python3 extensions/cooperative_moe/prepare_profile.py --stock overlay/exl3.py --artifacts ~/.cache/vllm-glm53-flash/cooperative_moe --runtime-directory /root/.cache/vllm/cooperative_moe --output <file>`.
Without a cooperative overlay, `start.sh` uses `overlay/exl3.py`, which already contains the #233 BF16 path.

## Evidence (Measured, TP2)

1. **Overnight 09-26b** (`logs/overnight-20260926b-tp2-f1-p1c-compact-20260925T184618Z/REPORT.md`, frozen gates, A/B/A,
   10-run W-DEC, paired bootstrap CIs):
   - FP8=all vs production (A2/A3): W-DEC GEO ms/cycle −4.67 % [−5.11, −4.06] / −4.91 % [−5.42, −4.44], 5/5 workloads;
     tok/s +5.2 / +4.3 %; W-AGT, W-PRE and C4 unchanged or better.
   - KDA BF16 at 11 GiB vs an 11 GiB stock baseline (Ac1): W-PRE 16k −12.06 %, 64k −12.61 % (CIs exclude 0); W-AGT
     −3.7…−9.8 %; memory gate passed (0 NV_ERR_NO_MEMORY; every workload minimum within 0.33 GiB of 14 GiB stock).
   - Frozen verdicts there were REJECT for both, only because the W-NUM screen (`tests/bench_numerical_calibrated.py`)
     exited 1. The same screen also flagged its own held-out stock control. Also reported: tokens/cycle (FP8=all) and
     hash-map decode (KDA BF16) just outside a narrow A-A band, with CIs spanning 0.
   - The one NVRM burst of that night (FP8=all, 20:34Z) coincided with 4 concurrent outside requests. The clean re-run
     had 0 lines and a higher head minimum than stock.
2. **Numerics study** (`logs/numerics-study-20260926-20260926T071116Z/RESULT.md`, gates frozen before the first
   capture): 3 independent stock boots × 2 `llm-quality-suite kl_panel` captures (prefill-only prompt logprobs on 20.7k +
   23.6k tokens of code, 12k prose, and two agentic tool-call tails), with a leave-one-boot-out stock control.
   - Stock control: NOT-WORSE **PASS**.
   - Both candidates: **NOT-WORSE** on every item. ΔNLL vs stock was −0.004…+0.0005 nats/token on text, with a tolerance
     of max(stock range, 0.005).
   - EQUIV was inconclusive: the stock boots themselves miss a max-of-pairs band by ≤ 3 %.
   - Interpretation: FP8=all's text KL is 25–30 % above every stock pair (a real, small numerics change that does not
     predict text worse); KDA BF16 sits at the stock edge.
   - The W-NUM screen flagged a held-out stock boot again, so it should not be used as a gate at 2 captures/arm.
3. **Adoption confirmation** (`logs/adopt-f1-pc-20260926T084218Z/`):
   - Both ranks verified: `dense fp8 groups: all`, overlay `895e5269`, 34 `kda bf16-large-m retained` lines per rank,
     KV 1,233,779 tokens = 1.45× at 850k, 0 NVRM lines.
   - Against the 09-26b stock arms (different day, not a same-window bracket): decode ms/cycle −4.4 / −4.6 %, tok/s
     +4.8 / +3.9 %; W-PRE −11.5…−12.1 %; W-AGT −3.6…−10.0 %; probe PASS; head MemAvailable ≥ 6.2 GiB under load.

## Costs and conditions

- **KV capacity −21.5 %** (1,572,073 → 1,233,779 GPU-KV tokens, a max_concurrency × max_model_len figure). An 850k
  request still fits (1.45×).
- The 11 GiB pool is sized for **compact draft pages** (`GLM53_DRAFT_KV_COMPACT=1`, the launcher default under
  dflash). With COMPACT=0 it holds only ~690k tokens and the boot refuses 850k; use 14 GiB there.
- **Retained memory:** the KDA BF16 copy costs +3.3 GiB/rank. The 3 GiB freed from the KV pool pays for it
  (AGENTS §5.6: memory and latency are evaluated together).
- Not a task benchmark: code pass@1 was not run. Tool-calling (17 × 3) was 45–47/51 for every arm, stock included.

## Rollback

`logs/adopt-f1-pc-20260926T084218Z/rollback.sh` restores `.env` `c3c3aba1` byte-for-byte and restarts. In the repo,
revert this commit.

## Related finding: compact draft pages (09-26b K0)

**Measured:** compact draft pages account for none of the 18.7 % H45 S3.7k agentic TTFT gain of the f497020 launcher
bundle; they cost 6–7 % on that cell. They do prevent a 2.4× replay-clamp slowdown on H62 S0.8k (2 clamps/run without
them), so they stay on.

## TP=3

Nothing here was measured at TP=3.
- The measured cluster's `.env.tp3` pins `GLM53_DENSE_FP8=dense,kda`, its own KV pool (35 GiB), and already sets
  `GLM53_KDA_BF16_LARGE_M=1`. These are cluster-local values, not all template defaults.
- `start-tp3.sh` clears the shared TP2 KDA-retention flag before reading `.env.tp3`, so fresh TP3 setups keep
  retention off. Set `GLM53_KDA_BF16_LARGE_M=1` in `.env.tp3` or the caller environment to opt in.
- `start-tp3.sh` unsets the TP2 `EXL3_OVERLAY_HOST` because the TP2 cooperative adapter is the wrong ABI for TP3.
- **Code-verified:** the #233 path lists the TP3 shape `8726×4096`.
- Rolling FP8=all into TP3 needs its own A/B/A (different per-rank shapes, and every rank must agree on the FP8 set
  or FULL capture hangs), plus the same stock-controlled numerics check.
