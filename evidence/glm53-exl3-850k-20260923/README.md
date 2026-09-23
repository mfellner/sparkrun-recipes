# GLM-5.3 refresh harness — 2026-09-23

**Functional acceptance PASSED: 35/35 checks on the final unchanged workload.**
The historical evidence remains unchanged. This package does not establish
throughput, a completed 850K request, or general model quality.

## Accepted run

- Workload: `sparkrun_3f2be18c41effca0_13351defc50c`; run ID `25c05299e4881ea2`.
- Config: `expected.json`, SHA-256 `177796ddfc9af8844321ea589a99990ab09b33fef33cee5e8f2d8804ef462baf`.
- Recipe: `887063a8e959936ad628367ca4166911da3b6c63abf0b82422d1f24b73688610`.
- Mod manifest: `fdd0866e185a1cde62b5d8f0bb2511f8404cb2d260d418c3d8a6fce972d169fb`.
- Receipt manifest: `a0a72ab5f84af7760bb84b4feb0950b99f89a76abb2731228b7d1d378237cd51`.
- `receipts/` contains the actual launch log/receipt, before/after rank captures,
  acceptance JSON and lossless HTTP journal. Full semantic verification passes.
- 110,035 prompt-token retrieval completed (warm prefix); 4,096 visible completion
  tokens completed. These are not full 850K capacity or performance measurements.
- APC: cold request reported 0 cached tokens; identical replay reported 14,336.
- Both worker processes map the pinned NCCL 2.30.7 library, preserve recipe
  environment, and remain in the captured serving lineage. Installed InstantTensor
  is 0.2.0. Both runtime overlay gates pass; E3 grouped execution counters are positive.
- Both kernel journals are exactly `-- No entries --` from readiness through
  acceptance. Full serving logs contain no verifier-defined fatal signatures.
- `--enable-prompt-tokens-details` is a local observability adaptation to expose
  actual per-request APC counts. `SPT_NOENV=1` preserves real worker environment.

Reverify from the recorded deployment checkout `/home/max/sparkrun-recipes`:

```bash
uv run --with pyyaml python evidence/glm53-exl3-850k-20260923/verify.py \
  --config evidence/glm53-exl3-850k-20260923/expected.json \
  --expected-config-sha256 177796ddfc9af8844321ea589a99990ab09b33fef33cee5e8f2d8804ef462baf \
  --out-dir evidence/glm53-exl3-850k-20260923/receipts
```

The config retains absolute deployment input paths intentionally; an exported
review tree must additionally byte-compare its recipe/mod to those pinned inputs.
Do not rewrite the config paths or historical receipt bindings to claim portability.

## Qualification and failed attempts

`qualification/` preserves the full CPU/APC run and both-rank GPU qualification.
The CPU suite passed 350 tests plus 70 subtests, with five locale-only skips;
scheduler standalone passed 30; real-vLLM APC no-store passed 186 checks, including
99 Part C checks. Both adapted full GPU gates passed. Their earlier recipe/mod
hashes differ from final only by documented observability argv/documentation
changes; executable overlay bytes are unchanged. `detokenizer-replay.json` records
four actual production-detokenizer tests with 21 boundary cases in the pinned
no-GPU image, including targeted RED controls. This is not engine/model EOS testing.

`diagnostics/` preserves failures rather than silently retrying them away. The
first two launches failed dgx02's 0.85 startup-memory gate, including after clean
cache release. The operator restarted an abnormally large polkit service before
successful loading; no memory-profile reduction was used. The first functional
attempt failed the unconstrained literal-stop probe; repeated probes confirmed
premature EOS or extra prose. The revised, explicitly constrained stop contract
below passed every predeclared repetition on a fresh launch. The original
unconstrained-generation issue remains unqualified, not declared harmless.
The first APC probe also lacked cached-token reporting until the observability
flag was added. Diagnostic files are not accepted-run evidence.

## Trust boundary and coverage

- `prepare.py` accepts independently reviewed **recipe SHA-256, mod manifest
  SHA-256, upstream source revision, immutable image reference, image config
  digest, and loaded NCCL path/version/SHA-256**. It checks the actual local
  recipe/mod, derives exact rank argv with the hash-pinned historical shell/YAML
  parser, and pins every Python file in this directory. Keep the resulting
  config digest outside the receipt directory; every subsequent command requires
  it. Re-preparation after a code/recipe change requires renewed review.
- Host order is fixed: rank 0 `192.168.178.47`, rank 1 `192.168.178.46`.
  Launch/run IDs are fresh inputs, not historical constants.
- Defaults fail closed: `fair`, `instanttensor`, explicit TP2 KV allocation
  `15032385536` bytes, 850000 configured context, C4, compact draft off,
  `DEFAULT_MAX_NEW_TOKENS=65536`, APC on, `media.invalid`, image=4/video=0.
  Remaining environment and argv values are derived from the pinned recipe.
- Canonical requests preserve exact C4 markers and overlap, the historical
  110K retrieval prompt (usage must substantiate the tier), direct/proxy exact
  completions, image quadrants/OCR, remote-media and inline-video rejection,
  tool calling, reasoning-open client stop suppression, and thinking-disabled
  literal stops. The disabled-thinking probe predeclares ten requests named
  `direct_thinking_disabled_stop_0` through `direct_thinking_disabled_stop_9`,
  each binding `ignore_eos: true`, `structured_outputs: {regex: 'BEFORE Question: AFTER'}`,
  `stop: ['Question:']`, and `enable_thinking: false`. Every receipt must contain
  exactly `BEFORE ` (including the trailing space), `finish_reason: stop`, and
  `stop_reason: 'Question:'`. All ten are required: no trimming, EOS substitution,
  retry-until-pass, or best-of-ten selection. The regex fixes the generated text
  and `ignore_eos` prevents early EOS from bypassing the literal-stop exercise.
  Previous unconstrained early-EOS results were not stop-policy proof and remain
  failed evidence, not reclassified passes. This constrained probe tests literal
  stop handling only; it does not establish unconstrained instruction following,
  general stop reliability across prompts, or full model quality. Preserve prior
  failed run directories unchanged and prepare fresh pinned config/receipts for
  this revised contract. New cases cover `tool_choice: none`, strict JSON schema with
  reasoning, visible long output exceeding 2000 completion tokens, omitted
  completion limit, and repeated-prefix APC with a reported cache hit.
- The omitted-limit request is deliberately short. Its successful completion
  plus both processes' environment and the installed-patch gate checks the
  default configuration/normal API path; **it does not empirically exhaust a
  65536-token omitted-limit request**. No claim of a measured boundary is made.
  Mamba alignment, fair-v5, tool-choice suppression, and omitted-default patch
  installation are checked by the pinned mod's `verify_runtime_patch_state.py`
  on both ranks; retain the mod's separate CPU regression results as well.
- Before/after snapshots bind exact SSH command/host, rank labels, container ID,
  image manifest/config identity, boot ID, container start, raw `docker top`,
  host/container PID joins, process start ticks, effective recipe environment,
  and the NCCL library **mapped by the live TP worker**, not just an installed
  package. Installed InstantTensor version is captured on each rank and checked
  against its explicit review pin (upstream 0.2.0). No optional fast-MoE symbol is
  required by this harness. The head API listener must belong to the selected serving PID; the
  headless worker must not own an API listener. Process/container identity must
  remain unchanged across acceptance.
- Full serving logs, runtime patch output, installed mod bytes, readiness
  timestamp, and kernel journal covering readiness through acceptance are
  captured with reconstructible commands. Missing/failed/redirection receipts,
  fatal logs, Xid/OOM/NV_ERR_NO_MEMORY, model/route substitution, missing tests,
  canonical request/fixture substitution, and timestamp gaps fail verification.
- The proxy checks prove response identity and semantics at the selected proxy
  URL. This compact harness does **not** certify the proxy supervisor lineage,
  every other model, RDMA throughput, thermals, fabric utilization, or measured
  850K capacity. Those are separate operational checks, not implied passes.

Historical request builders/fixture generators and command parsing are imported
only after their source hashes match the reviewed historical files. No old
receipt is ever reused as new live evidence. CPU tests explicitly label their
synthetic data and confine it to pytest's scratch directories.

## Operator commands (not executed by preparation)

Run from `/home/max/sparkrun-recipes`. Python needs PyYAML; `uv run --with pyyaml
python` supplies it without changing the system interpreter. The examples below
use Bash arrays. Supply pins from the completed **independent** recipe/image
review, not by blindly refreshing a manifest to make a failed check pass.

```bash
umask 077
H=evidence/glm53-exl3-850k-20260923
S=/home/max/.hermes/cache/scratch
# SparkRun 0.3.6: exactly 16 lowercase hex intent + 12 lowercase hex placement.
# Choose a fresh placement token and run ID; variable-length suffixes are invalid.
: "${LAUNCH_ID:?sparkrun_<16 hex intent>_<fresh 12 hex placement>}"
: "${RUN_ID:?fresh 16 lowercase hexadecimal characters}"
C="$S/glm53-$RUN_ID-expected.json"
R="$S/glm53-$RUN_ID-receipts"

uv run --with pyyaml python "$H/prepare.py" \
  --recipe recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml \
  --mod-dir mods/glm-5.3-flash-exl3-upstream-850k \
  --recipe-sha256 "${REVIEWED_RECIPE_SHA256:?}" \
  --mod-manifest-sha256 "${REVIEWED_MOD_MANIFEST_SHA256:?}" \
  --source-revision 3f2be18c41effca0b4b2c6a65f0b24a7a9f39567 \
  --image "${REVIEWED_IMAGE_REFERENCE:?immutable @sha256 reference}" \
  --image-id "${REVIEWED_IMAGE_CONFIG_DIGEST:?sha256 prefixed config digest}" \
  --nccl-path "${REVIEWED_NCCL_PATH:?canonical loaded library path}" \
  --nccl-version "${REVIEWED_NCCL_VERSION:?}" \
  --instanttensor-version 0.2.0 \
  --nccl-sha256 "${REVIEWED_NCCL_SHA256:?}" \
  --launch-id "$LAUNCH_ID" --run-id "$RUN_ID" --cluster vacation-pair2 \
  --direct-url http://127.0.0.1:8000 --proxy-url http://127.0.0.1:4000 \
  --out "$C"
# Record the printed digest in the review record, then set this external pin:
: "${EXPECTED_CONFIG_SHA256:?digest printed by reviewed preparation}"
ARGS=(--config "$C" --expected-config-sha256 "$EXPECTED_CONFIG_SHA256" --out-dir "$R")

# DEPLOYMENT SIDE EFFECT: run ONLY after explicit launch authorization.
# Do not separately launch and then manufacture a retrospective receipt.
uv run --with pyyaml python "$H/launch_receipt.py" "${ARGS[@]}" --execute-launch

# Wait for the in-container /tmp/glm53-postready.ok marker and separately
# verify proxy discovery. The before collector fails if readiness is absent.
uv run --with pyyaml python "$H/capture_runtime.py" "${ARGS[@]}" --phase before
uv run --with pyyaml python "$H/acceptance.py" "${ARGS[@]}"
uv run --with pyyaml python "$H/capture_runtime.py" "${ARGS[@]}" --phase after
uv run --with pyyaml python "$H/verify.py" "${ARGS[@]}" --write-manifest
uv run --with pyyaml python "$H/verify.py" "${ARGS[@]}"
```

`launch_receipt.py` refuses a nonempty receipt directory and requires its explicit
execution switch. Other producers refuse to overwrite completed receipts. The
HTTP journal is flushed per response; an error/timeout is preserved, never
converted to a successful response. A partial run cannot pass. After a timeout,
check that the server has canceled/drained it before another attempt. Use new
reviewed IDs/config and a new output directory for a new accepted run.

SSH must already trust both hosts and permit Docker inspection, host `/proc`
inspection, and kernel journal reads without password prompts. The selected
security contract is root-user, nonprivileged, host network/IPC, IPC_LOCK,
no-new-privileges, and label disabling, matching the reviewed SparkRun path.
A changed executor/security shape fails rather than being silently accepted.
Clock synchronization is an operational prerequisite for cross-host journals.

**Publication:** output directories are private. Full logs/argv can contain
sensitive material; this harness intentionally does not run the old, fixed-command
redactor over new commands. Audit raw files before publication, and do not call
these files sanitized. Preserve immutable config/harness/recipe/mod provenance
and the resulting receipt-manifest digest with the review. Hashes and semantic
checks are tamper-evident consistency checks, not signed host attestation.

## CPU verification

```bash
TMPDIR=/home/max/.hermes/cache/scratch \
uv run --with pytest --with pyyaml python -m pytest tests/test_glm53_refresh_evidence.py -q
```

No SSH, deployment, GPU/model requests, staging, commit, or push is performed by
these tests. CLI help, historical positive receipts, explicit synthetic unit
bundles, refreshed-manifest negative controls, and mocked HTTP failures exercise
the harness without representing synthetic data as deployment evidence.
