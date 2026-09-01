# GLM-5.3 Flash EXL3 upstream 1M runtime mod

This SparkRun mod vendors the exact runtime overlay set from:

- Repository: `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`
- Commit: `c190db1ae17ba8dff20129ed1f308d10c63cf37d`
- Pinned image expected by the recipe: `ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks@sha256:4f30fba4248ed7d78dddda37d9ffc0fea5cc8c567c5dd98042ad808efdde9791`

`run.sh` executes inside every rank container before vLLM starts. It verifies every vendored file against `SHA256SUMS`, validates the selected GID on every SparkRun-selected HCA, checks the compiled E2 symbols, installs the matching EXL3 Python overlay/chat template, and applies the video, DFlash2, mixed-prefill, reasoning-stop, APC, XGrammar, K-pool, spinwait, guarded indexer-workspace, and optional-off ABLIT corrections in upstream order. Every patcher is idempotent and fails on source-anchor drift. The complete source/static suite runs separately against the exact bundle; live per-rank diagnostics prove kernel-tier TP2 execution. Avoiding repeated test-framework imports inside every pre-serve container preserves vLLM's startup-free unified-memory margin. The effective profile sets `EXL3_FAT_KERNEL=1`, `MAX_NUM_BATCHED_TOKENS=7168`, `GLM53_INDEXER_WORKSPACE=rightsize`, and `GLM53_SPINWAIT_MS=stock`; rightsize is upstream's opt-in legal-maximum formula and is covered by its exhaustive chunk-equivalence gate.

The adjacent `patch_suppress_stops_multitoken.py` makes stop handling deterministic for this default-thinking profile: while the recipe guard is enabled, client-provided stop strings are ignored and EOS/`max_tokens` govern completion. `GLM53_SUPPRESS_STOPS_IN_REASONING=0` restores stock stop behavior for deployments that require client stops. Its regression gate covers the patched stop-check seam plus enabled and opt-out paths.

`serve_wrapper.sh` receives SparkRun's appended native-distributed flags, launches `postready_gate.sh` only for `--node-rank 0`, then execs vLLM; workers exec vLLM directly. The rank-0 gate allows up to 3,600 seconds for readiness before running `boot-shape-warmup.sh` plus an exact semantic gate. Upstream treats the warmup sweep as non-fatal; this publication intentionally promotes any gate failure to fatal, signals the serve process group, waits, escalates to SIGKILL, and verifies termination. `test_postready_termination.sh` is a TERM-resistant negative control. This avoids SparkRun 0.3.6's optional 240-second post-hook readiness lifecycle while remaining fail-closed.

`SHA256SUMS` covers every mod file, including `run.sh`, this README, helpers, local follow-up, and all vendored upstream payloads; only `SHA256SUMS` itself is excluded.

Do not edit vendored patch/test files locally. Refresh only from a newly audited exact upstream commit, regenerate checksums, repeat disposable-container application on both ranks, and rerun the live 1M acceptance suite.
