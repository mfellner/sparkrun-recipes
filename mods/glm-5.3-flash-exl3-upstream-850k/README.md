# GLM-5.3 Flash EXL3 upstream 850K runtime mod

This SparkRun mod vendors the **complete tracked source tree** from:

- Repository: `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`
- Commit: `3f2be18c41effca0b4b2c6a65f0b24a7a9f39567` (258 files)
- License: AGPL-3.0; `upstream/LICENSE` is authoritative and
  `upstream/LICENSE.MIT` retains the earlier license text.
- Default-profile image: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d`
  (public `exl3-instanttensor` tag resolved on 2026-09-23).

## Profile and image boundary

The public arm64 image supplies InstantTensor 0.2.0 and the existing E3 grouped
fat-expert extension. It is **not** an exact build of the latest Dockerfile:
its compiled EXL3 layer predates optional thin-decode kernels. The default
`GLM53_EXL3_MOE_FAST=0` path does not require those symbols. Enabling FAST=1,
or using the optional upstream long-coding/cooperative profile, requires a
separately qualified image and GPU tests; a Python overlay cannot add compiled
kernels. No such opt-in is selected here.

The TP2 profile keeps 850000 context, utilization 0.85, sequences 4, batch tokens
7168, DFlash2 k=7/draft TP2, FP8 target KV, E3/32 fused temporary rows and
right-sized indexer workspace. New defaults are:

- `--load-format instanttensor` and `--kv-cache-memory-bytes 15032385536`
  (14 GiB). The reservation comes from upstream `.env.example`; `start.sh`
  alone warns rather than installing it in an existing dotenv file.
- Fair-v5 mixed-prefill scheduling: chunk 256, share 0.30, max interval 2000 ms,
  max step 2000 ms, max chunks 1; Mamba chunk alignment and state-lifetime fixes.
- Hardened hybrid APC/boundary support, client-requested no-store support ON
  (`GLM53_APC_NO_STORE=1`, requests remain store-enabled unless they opt out),
  and KV-capacity logging ON.
- `DEFAULT_MAX_NEW_TOKENS=65536` changes only the omitted-request fallback for
  chat and legacy completions. Explicit client limits still win, subject to
  independent model/platform caps. It is not a hard output or admission cap.
- `tool_choice:none` decode-time masking with the glm47/glm45 parser setup.
- Per-image token budget 2048 and processor cache 1 GiB.
- Refreshed boot warmup includes 3584/7168/14336/65536-token prefill rungs.

Compact draft KV, thin decode, KDA BF16 large-M, dense FP8, adaptive-k,
cooperative MoE, runtime ABLIT and cache-reset endpoints remain OFF. Sparse
retention and NCCL channel overrides remain unset, retaining stock policy.
The target snapshot remains `024db9f7e9871e8efdf21538ba55af7442be3cd5`
(serving-payload-equivalent to upstream `25a44fdb...`); draft remains upstream's
explicit `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, not moving HF main.

## Runtime composition and local boundaries

`run.sh` applies upstream `GLM53_OVERLAY_ORDER` in exact order. The only local
insertions are the scoped stop-policy repair immediately after upstream stop
suppression and the E3 execution marker immediately before dense-FP8 installs
the EXL3 module. The existing TP2-local ABLIT hook installer remains last.

`verify_runtime_patch_state.py` validates **final composed source**, including
all fair-v5 hooks, Mamba changes, compact-off drafter code and worker split
guard, boundary/replay coordinator plus per-group retention, no-store's three
files, capacity diagnostics, output defaults, tool control and cache routing.
It also retains exact stop-policy, E3 marker, video hook, indexer, kpool,
xgrammar, spinwait, dense-off and ABLIT checks. Missing, duplicated, drifted
and stale snippets fail closed; stock anchors embedded inside canonical
replacements have precisely counted allowances. Hybrid and capacity helpers
also use upstream's owned-binding checks. Two legacy drafter **comment-only**
paragraphs survive in the public image; all executable bytes around them are
still checked. Fair-v5's generated helper needs an importlib-registered module
for `inspect.getsource`, rather than a transient runpy namespace.

The CLI always checks live video-import ownership. CPU fixture tests can call
the Python API with runtime imports disabled, but that is not a launch gate.
The mod checks InstantTensor's installed version and six E3 symbols before
serving; it does not require the optional thin symbol when FAST=0.

The API stays on port 8000 with stable alias `GLM-5.3-Flash-EXL3`. Immutable
snapshots and offline loading remain. Remote media is restricted to
`media.invalid`; only inline images are supported, with the intentional local
limit `{"image":4,"video":0}`. Keep inference/distributed control ports and
proxy port 4000 on a trusted firewalled LAN. Cache-reset routes stay disabled.

`serve_wrapper.sh` retains process-group supervision and cleanup on both
ranks. Rank 0 retains the fail-closed post-readiness gate: up to 720 API polls
with a 5-second request timeout and 5-second sleeps, followed by refreshed
upstream warmup, an exact C4 semantic wave and the existing 110K retrieval shape.
This is a polling budget, not a 3600-second overall deadline.
The gate terminates vLLM on failure. The local stop policy suppresses client
stop strings only while reasoning is open, resumes matching after `</think>`,
and leaves thinking-disabled requests unguarded.

## Verification and prerequisites

Live qualification status and exact run bindings are maintained in
`evidence/glm53-exl3-850k-20260923/README.md` and the repository README.
Prior 20260913 results (including the former 114-test source-suite receipt)
are historical, not proof of the new image/profile. No old evidence is rewritten
by this mod.
`run.sh` does not run the complete upstream test suite during pre-launch.

CPU-only checks exercised during this refresh:

- Complete exact-commit source-tree parity and local test-first contracts.
- Actual latest runtime patch sequence in an isolated, no-device, no-network
  disposable pinned image, followed by static final-state verification and a
  second complete application with byte-identical runtime source.
- Final-source missing/duplicate/stale-fragment negative controls; local
  stop-policy and E3-marker self-tests; fair scheduler (30 tests), restart,
  hybrid-prefix and adaptive-k standalone tests.
- Earlier 768 MiB full-suite attempts exited 137 (not passes). After the
  authorized model unload, the complete 12 GiB offline qualification exited 0:
  **350 passed, 5 skipped, 70 subtests passed**. All five skips are unavailable
  comma-decimal locales in `test_instanttensor_kv_note.py`; no scheduler tests
  skipped. The standalone fair scheduler additionally passed all 30 tests.
- Mandatory real-vLLM APC no-store composition passed **186 checks**, including
  **99 Part C checks with zero failures** and explicit seven-group KpoolTail /
  Mamba / drafter layouts. The pristine-base per-group retention gate passed,
  including both composition orders, AST equality and idempotence. The actual
  complete runtime patch stack passed twice with byte-identical final sources.
- Local adapter regression tests passed 2 tests. The unchanged NIC module went
  from 4 failed / 9 passed without the adapter to 13 passed with it. Exact
  upstream archive parity remained 258 files. Raw session logs are retained at
  `/home/max/.hermes/cache/scratch/glm53-full-compatibility-final-20260923.log`,
  `glm53-fixture-red-20260923.log`, `glm53-fixture-green-20260923.log`, and
  `glm53-source-parity-final-20260923.log` in the same scratch directory; these
  session-local logs are not a committed evidence bundle or live acceptance.

Full offline qualification runner:

```bash
python3 verify_upstream_source_parity.py
GLM53_VLLM_SRC=/path/to/pristine-vllm bash run_upstream_compatibility_suite.sh
```

Both pinned public and pristine base images must already be available; the
runner does not pull. `GLM53_VLLM_SRC` must contain `vllm/v1/` files from vLLM
`487ecf187d3dfe74d2cf6119a92881dba403c219`; the five exact hashes are in
`upstream/tests/test_draft_kv_compact.py`. The runner supplies the source-root
Python path, excludes the two explicitly GPU-only optional tests, runs the
standalone tests, applies/verifies the actual mod twice, then requires
`GLM53_REQUIRE_VLLM=1 GLM53_REQUIRE_COMPOSITION=1` for no-store and the pristine
base-image APC gate. Default memory/swap cap is 768 MiB. Only after separately
unloading workloads and checking host headroom may the operator explicitly
raise `GLM53_TEST_MEMORY`; a killed or skipped mandatory gate is not acceptance.

The local `upstream_fixture_adapter.py` plugin repairs test inputs only: the
NIC fixture receives its existing temporary placeholder for the two missing
Mamba overlay paths, without replacing the real preflight or its assertions.
Explicit invalid inputs remain invalid. Pytest also runs the scheduler module's
unchanged source-installation helper, so its three running-loop tests execute
rather than skip. The standalone scheduler gate still repeats all 30 tests.

`run_upstream_gpu_compatibility.py` is a separate GPU qualification wrapper,
not a runtime change. It replaces only the exact obsolete eight-line compact-KV
assertion block inside `_check_dflash2` in memory, requires complete current
runtime verification and upstream's compact-block geometry/OFF-to-64 test, and
retains every other upstream assertion and GPU routine. Changed or duplicated
adapter anchors fail closed. Run it after `run.sh` in an isolated GPU container;
it refuses `EXL3_SELFCHECK_GPU=0`. A CPU adapter regression pass is not evidence
that this full GPU wrapper has run.

A small standalone composition check (no model/cache mounts):

```bash
docker run --rm --network none --memory 768m --memory-swap 768m --cpus 1 \
  --entrypoint python3 -e GLM53_DISPOSABLE_CPU_TEST=1 \
  -e DEFAULT_MAX_NEW_TOKENS=65536 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/review/mod:ro" \
  ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d \
  -S /review/mod/test_runtime_patch_composition.py
```

Before deployment: validate/dry-run SparkRun; verify image IDs and complete
snapshot inventories on both ranks; check CX-7/GID/fabric health; stop the old
model only through an authorized transition; check host memory/earlyoom and
vLLM admission headroom; run image/GPU qualification and fresh direct/proxy,
reasoning-stop, tool-none, vision/zero-video, concurrent, long-context and
kernel/runtime acceptance. Do not inherit old capacity/performance claims.

`upstream/scripts/boot_candidate.sh` remains an archival upstream experiment
that masks launch failure. It is vendored byte-for-byte but is not invoked by
the runtime call graph; do not use it as a launcher.

`SHA256SUMS` covers every mod file except itself (generated Python/test caches
are not artifacts). Refresh upstream only from an audited immutable commit,
retain its licenses, regenerate the complete manifest, and repeat qualification.
