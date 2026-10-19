# Qwen3.8-27B / SparkRun 0.4.1 compatibility

This is a new single-host compatibility run of the migrated recipe, not a
continuation of the older 0.3.6 LAN process. The tested model has been unloaded.

## Freshness

`upstream.json` preserves successful live HTTP responses and Git HEAD lookup:

- Mia main: `c11ff138871cf43d12898d79dbfa4dc952d17ca4` (unchanged).
- Both GHCR `latest` and `v0.6.0-fb91a0019fdc` resolve to the recipe's manifest
  digest `043d192b9e1725f55c5439fc9672055de891433e5b31448ed8963918e3c460e9`.
- Target HEAD `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` and draft HEAD
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` match the retained pins.

Mia still selects TensorFold 0.6.0 with patch hash `fb91a0019fdc`. A newer
standalone TensorFold release is not an approved replacement for Mia's patched
image. No engine, checkpoint, image, profile, or sampling upgrade was needed.

## Migration and measured result

Recipe SHA-256: `40f80456af390a49f4a84684e71791941da89680cf1f349b00ca8f68d24678e8`.

- Removed the ignored top-level recipe name.
- Explicit `vllm-distributed` solo lifecycle shim (TensorFold is the engine).
- Target `{model}` is resolved by SparkRun to its validated immutable snapshot.
- Draft path uses `$HF_HUB_CACHE`; its pre-launch checksum verification remains.
- Compilation caches use SparkRun's managed `/cache/runtime` mount, keyed by
  image and model. No old caches deleted; automatic pruning disabled.
- The original security posture and full 78 GiB/8-stream/262144 profile remain.

`launch.json` is a successful real CLI launch with `--no-follow` and the default
loopback bind, return code 0, including SparkRun's streaming inference readiness.
The stored launch receipt hashes the exact recipe and all mod files.

`runtime-before.json` and `runtime-after.json` bind the same container ID,
listener PID/start ticks, engine source hash, actual argv/env, mounted cache,
security settings, prepared model path and job recipe to the HTTP test interval.
SparkRun now invokes Docker by the immutable image configuration ID; the job's
`effective_container_image` and upstream manifest bind that ID to the recipe digest.

`direct/` contains real request/response receipts: exact chat, eight overlapping
HTTP clients, red/blue image understanding, typed tool arguments, and exact needle
retrieval from **41,443 API prompt tokens**. The existing fail-closed semantic
verifier passed all 12 cases and rejected four corrupted-evidence controls:

```bash
python scripts/qwen38/verify.py \
  evidence/qwen38-tensorfold-sparkrun041-20261019/direct --negative-controls
```

`compatibility.json` records the semantic result and additional checked runtime
bindings. The startup log reports the unchanged 78.0 GiB pool / 2,555,904 token
capacity. These are configuration/allocation figures, not a full-context test.
`logs.json` preserves the complete server log plus identical repeated bounded
kernel-log queries; no matched fatal engine/CUDA/Xid/OOM signatures were found.
`cleanup.json` records stop, proxy refresh, two-host empty SparkRun inventory,
empty proxy model list and empty local GPU compute-process inventory.

## Limitations and unsuccessful attempt

- `lan-attempt-runtime.json` is **diagnostic only**, never accepted as a passed
  launch: the first `--no-follow` launch bound only `192.168.178.47`, while 0.4.1's
  local readiness observer probed `127.0.0.1`. The enclosing foreground tool hit
  its 600-second timeout and killed the wrapper before it could write a launch
  receipt. The server was alive and healthy, but that does not make the CLI
  attempt pass. It was stopped before the separately accepted loopback launch.
- LAN-only users on 0.4.1 must use `--no-ready-wait` and separately verify HTTP and
  semantic readiness against their chosen LAN address. No wildcard bind was added.
- Validation retains one false-positive warning, `unpinned-model-revision`:
  vLLM-specific lint expects `--revision`, whereas TensorFold receives a validated
  absolute snapshot path. The runtime receipts prove the exact pin actually used.
- Proxy inference was not requalified. Historical 0.3.6 proxy evidence is not
  reused as acceptance for this process.
- No 262K near-limit inference, eight full windows, video/50-image capacity,
  throughput parity, numerical quality parity, or sustained uptime claim.
- Only dgx01 executes this single-node recipe. Other fleet nodes are not implied
  by this compatibility result.

`static-before-migration.json` is the original 0.4.1 lint/dry-run output;
`static.json` is the migrated recipe's validation, dry-run, alias tests and shell
syntax check. Separate pytest logs retain focused and baseline-comparison results;
repository-wide baseline failures are not silently described as a clean suite.
The older exact-recipe deployment verifier applies only at Git commit `01284da`.
