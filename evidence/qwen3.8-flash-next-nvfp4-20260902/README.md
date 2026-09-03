# Qwen3.8 Flash Next NVFP4 validation evidence

This directory preserves the preflight, launch, acceptance, telemetry, and failure evidence for the immutable SparkRun recipe `recipes/qwen3.8-flash-next-nvfp4-dual-spark-1m.yaml`.

## Accepted configuration

- Workload: `sparkrun_763b8e94b6d2bf09_27aa44aef83a`
- Controller cluster: `spark-ring3`
- Participating ranks: `dgx03` and `dgx04`
- Topology: exact two-rank tensor parallelism (`TP=2`)
- Served model: `qwen3.8-flash-next`
- Endpoint at acceptance: `http://192.168.178.48:8000/v1`
- Configured maximum model length: `1,000,000`
- Checkpoint revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`
- ARM64 image digest: `sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`

The initial TP2 acceptance result is `final-tp2-acceptance-896k.json`. It records passing deterministic chat, four-way concurrent requests, vision/OCR, structured tool calling, and retrieval from a request with `896,051` prompt tokens. `final-tp2-telemetry-896k.json` contains simultaneous rank telemetry captured during that long-context run.

After fail-closed readiness supervision and LAN-address/media-domain hardening, the exact final recipe was relaunched. `final-secure-postready.txt` contains its successful semantic gate receipt. `final-secure-live-status.json` is the SparkRun status snapshot for the two running ranks and idle `gx10` spare. `final-secure-acceptance.json` records a fresh direct matrix—deterministic chat, C4, inline-data vision/OCR, structured tool calling, and a 70,051-prompt-token retrieval—on the final process. `final-secure-proxy-regression.json` records fresh direct and LiteLLM-proxied exact completions for Qwen and the unrelated GLM service, two healthy proxy endpoints with zero unhealthy endpoints, and all five sparkDash nodes. `final-secure-rank-dgx03.txt` and `final-secure-rank-dgx04.txt` are the final container/rank audits. The expensive 896,051-token result belongs to the preceding process and is not presented as rerun after hardening.

`supervision-negative-test.txt` documents malformed-response, atomic/stale-receipt, receipt-path failure, and TERM-resistant-process controls. `gate-host-binding-regression-test.txt` verifies that the readiness gate uses the explicitly selected serving host rather than assuming loopback. `topology-negative-test.txt` preserves SparkRun 0.3.6's one-host solo fallback and proves the execution-boundary guard rejects it with exit 64 before vLLM starts; valid rank-0 and headless rank-1 controls pass.

The final guarded launch emitted bounded GB10 driver allocation-pressure notices during startup and the first accepted workload's graph capture: 65 on `dgx03` from 11:39:10–11:40:23 and 97 on `dgx04` from 11:39:07–11:40:22. Semantic readiness completed at 11:39:57, so a small bounded tail occurred while the fresh C4/vision/tool/70,051-token acceptance matrix exercised new shapes; that matrix completed at 11:40:47. Fresh journal scans after acceptance found zero further `NV_ERR_NO_MEMORY` notices and zero Xid/OOM-kill/reset events. The bounded notices are preserved rather than mislabeled as startup-only or silently discarded.

## Rejected topology

A three-rank trial was run so that all three physical members of `spark-ring3` could be tested. It reached NCCL initialization, then failed model construction because the vision tower has 16 attention heads and 16 is not divisible by 3. The exact failure is preserved in `tp3-failed-dgx03-serve.log` and `tp3-live-launch.txt`. The final recipe therefore enforces two nodes and leaves `gx10` as a prepared spare.

## Evidence classification

Files prefixed `final-tp2-` describe the accepted pre-hardening TP2 run; files prefixed `final-secure-` describe the accepted final hardened relaunch. `tp3-live-launch.txt` and `tp3-failed-dgx03-serve.log` preserve the rejected TP3 trial and do not count as acceptance. The repository-level `final-secure-audit-20260903.json` contains the final recipe hashes, bind scope, receipt semantics, runtime/log checks, and links to the final acceptance artifacts. The earlier broad socket-inventory audit was intentionally excluded from publication because it contained unrelated listener and overlay-network details.
