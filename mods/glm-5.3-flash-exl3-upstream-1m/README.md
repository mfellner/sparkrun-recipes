# GLM-5.3 Flash EXL3 upstream 1M runtime mod

This SparkRun mod vendors the exact runtime overlay set from:

- Repository: `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`
- Commit: `32db610d9207a42e2688a6994d3bfaf7af96eecb`
- Pinned image expected by the recipe: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58`

`run.sh` executes inside every rank container before vLLM starts. It verifies every vendored file against `SHA256SUMS`, validates the selected GID on every SparkRun-selected HCA, installs the latest stderr-only video overlay, upgrades the image's older GLM KV layout to padded DFlash2 slot-sharing, applies the mixed-prefill decode floor, reasoning-stop guard, hybrid APC correction, and XGrammar termination/reasoning backport in upstream order, then runs the upstream warm-restart, inline `MAX_NUM_SEQS` override, XGrammar, GPU, and CPU gates fail-closed. The upstream `python3 -S` launcher fix is retained and tested in `upstream/start.sh`; SparkRun's actual serve command uses static recipe JSON and never executes that shell command substitution.

The adjacent `patch_suppress_stops_multitoken.py` makes stop handling deterministic for this default-thinking profile: while the recipe guard is enabled, client-provided stop strings are ignored and EOS/`max_tokens` govern completion. `GLM53_SUPPRESS_STOPS_IN_REASONING=0` restores stock stop behavior for deployments that require client stops. Its regression gate covers the patched stop-check seam plus enabled and opt-out paths.

`serve_wrapper.sh` receives SparkRun's appended native-distributed flags, launches `postready_gate.sh` only for `--node-rank 0`, then execs vLLM; workers exec vLLM directly. The rank-0 gate allows up to 3,600 seconds for readiness before running `boot-shape-warmup.sh` plus an exact semantic gate. Upstream treats the warmup sweep as non-fatal; this publication intentionally promotes any gate failure to fatal, signals the serve process group, waits, escalates to SIGKILL, and verifies termination. `test_postready_termination.sh` is a TERM-resistant negative control. This avoids SparkRun 0.3.6's optional 240-second post-hook readiness lifecycle while remaining fail-closed.

`SHA256SUMS` covers every mod file, including `run.sh`, this README, helpers, local follow-up, and all vendored upstream payloads; only `SHA256SUMS` itself is excluded.

Do not edit vendored patch/test files locally. Refresh only from a newly audited exact upstream commit, regenerate checksums, repeat disposable-container application on both ranks, and rerun the live 1M acceptance suite.
