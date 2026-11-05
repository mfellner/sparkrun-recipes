# GLM-5.3 fffe64 refresh — bounded evidence (2026-11-05)

This is a small durable subset of preserved test output for the current recipe refresh. It is **not** live-service acceptance evidence and does not qualify the current recipe for production/GPU inference.

## Identity and scope

- Recipe: `recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml`
- Upstream source revision: `fffe64fc093f65fa8f033957e16b744dc266c2ec`
- Deployed workload reported by the operator: `sparkrun_58201f8c57f8af18_f50636d19f88` (identity not independently bound to these files)
- Recipe SHA-256 at evidence packaging: `971dad4074c325e5caaa4a1d0a06b145301343352bf116f1b4b8357a63434fd7`
- Mod manifest SHA-256 at packaging: `2ed3853d89eb26c5e83b5cb5f2ff73aef851db192f4b8a9b939747d23f74e20e`

## Preserved outputs

- `cpu/cpu-suite.log`: CPU/static composition suite transcript, ending `CPU_SUITE_EXIT_CODE=0`. Its own summary states 64 pytest modules passed, 0 failed, 7 had no pytest cases; GPU files were excluded. This is test evidence only, not runtime acceptance.
- `gpu-compat/rank0.log` and `rank1.log`: separate rank-labelled GPU compatibility probe output. Both include GPU kernel checks and `glm53 EXL3 overlay verify OK`; both explicitly say real-checkpoint E3 parity was skipped because the HF cache was not mounted. These are not records from the running workload, and do not prove its launch identity, model inference, API semantics, readiness, or health.
- `gpu-compat/manifest.log`: copied mod-manifest verification output (`OK` rows); it establishes only the source tree checked by that probe, not what bytes the live workload used.
- `SHA256SUMS`: hashes for the preserved evidence files and the recipe/mod-manifest identity above.

## Honest release boundary

The README's `fffe64` source revision, recipe configuration/pins, and explicit statement that the refresh lacks live qualification are supportable. The prior 2026-09-23 35-check qualification remains historical evidence for `3f2be18`, not the current profile. Do not describe the current refresh as live-qualified, fully GPU-qualified, full-context-qualified, or performance-qualified based on this package.

A separate read-only collection from the live service has since been captured in `live/`. It records the persisted job/recipe metadata, both rank container/process/overlay identities, bounded SparkRun logs and fixed-window kernel journal matches on both hosts, plus short direct and proxy API semantic/concurrency smokes. See `live/README.md` for scope and limitations. This does not prove 850k context capacity or clean sustained operation: no near-850k prompt was run, and the bounded journal contains a host OOM event not attributed to the GLM job. The runtime was not modified; no commit or push was performed.
