# GLM-5.3 Flash EXL3 upstream 850K runtime mod

This SparkRun mod vendors the complete tracked source tree from:

- Repository: `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`
- Commit: `1caea9a10b26ae93b88d08e82d1e7abb0dc45a42`
- Source license at that commit: AGPL-3.0 (the vendored `upstream/LICENSE` is authoritative; `upstream/LICENSE.MIT` preserves the repository's earlier MIT license text)
- Pinned image expected by the recipe: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:eecb36e14dc34c92d46827fde7b09f7e0bf27e27c426ece126376c02dea6cd2f`

The public image is the immutable arm64 E3 build published on 2026-09-07. It contains the compiled grouped fat-expert symbols added before the pinned source commit. Mia's subsequent adaptive-verification and dense-FP8 additions are pure-Python runtime overlays; `run.sh` installs those exact bytes from the pinned commit on every rank. Both options remain explicitly off, matching upstream defaults. The recommended source-revision profile enables the default E3 grouped tier, uses 32 fused temporary rows, `MAX_NUM_BATCHED_TOKENS=7168`, `GLM53_INDEXER_WORKSPACE=rightsize`, `GPU_MEM_UTIL=0.85`, and 850,000-token context. The older 1M/E2 recipe remains unchanged as rollback because upstream documents that E3's persistent scratch no longer fits 1M at utilization at or below 0.87 on this two-Spark geometry.

`run.sh` verifies the mod manifest, fail-closes each selected HCA on the exact GID index's `RoCE v2` type, sysfs netdev, and IPv4-mapped assigned address, installs the listed EXL3/chat-template/ABLIT payloads, applies the listed overlay installers, runs `verify_runtime_patch_state.py`, and imports the six required E3 extension symbols. These are the committed pre-serve gates; `run.sh` does not run the complete upstream test suite during pre-launch. `serve_wrapper.sh` supervises and verifies cleanup of the serving process group on every rank; only rank 0 also owns the post-readiness gate. The adjacent local `patch_suppress_stops_multitoken.py` scopes client-stop suppression to open reasoning, resumes matching after `</think>`, and leaves thinking-disabled requests unguarded.

Repository contract tests exercise the verifier adversaries, stop-policy behavior,
recipe wiring, and RoCE helper without launching a workload. Two separately named
release gates cover vendored source without misreporting upstream:

```bash
python3 verify_upstream_source_parity.py
bash run_upstream_compatibility_suite.sh
```

`verify_upstream_source_parity.py` downloads the exact `1caea9a10b26ae93b88d08e82d1e7abb0dc45a42`
GitHub archive and fails closed unless its relative file set and every file byte
match `upstream/`. `run_upstream_compatibility_suite.sh` is the **upstream compatibility suite**:
the full vendored upstream suite passes with **87 passed** and no deselections,
then `test_suppress_stops_multitoken.py` runs as an independently executable
synthetic host self-test. The live runtime gate invokes the same file with
`--production` against the exact pinned in-container detokenizer path. Both
commands must exit zero; neither permits an unparseable or partial result.

`upstream/scripts/boot_candidate.sh` is an archival upstream experiment that masks `start.sh` failure by continuing to append its exit status. It is vendored byte-for-byte for source parity but is not invoked by the recipe, `run.sh`, `serve_wrapper.sh`, or `postready_gate.sh`; the static recipe contract proves that runtime-call-graph exclusion. Do not use it as a launcher.

`SHA256SUMS` covers every mod file except itself. Do not edit files under `upstream/`; refresh them only from a newly audited exact upstream commit, regenerate checksums, rerun the clean-image source/GPU gates, and repeat live dual-rank acceptance.
