# Qwen3.8-27B TensorFold: final single-Spark acceptance

## Accepted scope

On dgx01 (192.168.178.47), SparkRun 0.3.6 launched workload
`sparkrun_143aaa9f3e323aa9_bcc74ffc19b7` using the recipe at SHA-256
`1e9aa10534862715249e75dff43d21fc9ebf45b615449f241902676caddbe628`.
The final container is
`0cbbce67f84eda06c1abe636cb24bc3bce6f8fee73c8d90dad4bf905ad9d7f6e`.

Final-process results:

- Direct `/v1/models`, `/health`, exact `SPARK_READY` chat.
- Eight concurrently issued requests, each returning its own exact code. The
  verifier requires all eight HTTP intervals to overlap; this is not a throughput
  benchmark or proof of eight simultaneously full-length windows.
- Inline deterministic PNG: red left half, blue right half; correct answer.
- Typed `get_weather` call with Berlin/celsius arguments.
- Retrieval of `SPARK_NEEDLE_7K2M` from **41,443 API prompt tokens**.
- LiteLLM chat, inline image and typed tool call on the same process; `/health`
  reports **one healthy endpoint and zero unhealthy endpoints**, routed to the
  exact private-LAN backend. Proxy binds loopback on port 4000.
- Runtime brackets show the same listener namespace PID/start ticks and host-PID
  mapping, exact container/image identity and unchanged configuration/mounts.
  Job bytes predate acceptance and contain the reviewed recipe and LAN override.
- TensorFold 0.6.0, transformers 5.17.0, PyAV 19.0.0; Torch distribution metadata
  `2.13.0a0+9186a08b2c.nv26.7.59513937`. Torch's runtime version string uses the
  alternative spelling `2.13.0a0+9186a08b2c.nv26.07`; these are different version
  presentations, not claims that the strings are identical.
- Startup log confirms **262,144 configured context**, up to **8 streams**, FP8
  prompts, drafts on and the **78 GiB pool / 2,555,904-token capacity**. Those
  configured capacities are distinct from the largest completed prompt above.
- Full serve log is independently hashed; two identical fixed-bound kernel
  queries cover launch to after final runtime capture. Neither log matches the
  verifier's fatal CUDA/Xid/OOM patterns. The kernel receipt includes SparkRun's
  host-wide `drop_caches: 3` pre-launch action. This is not a guarantee of future
  stability. No Docker healthcheck exists in the image.

## Inputs and failures retained

`launch.json` is the successful final-LAN launch. `runtime-before.json` brackets
all direct requests, `runtime-after.json` follows them, and `runtime-final.json`
follows the proxy suite. `logs.json` captures full serve and bounded kernel logs.
`direct/` and `proxy/` preserve canonical request dictionaries, raw HTTP body bytes
in base64, parsed responses, statuses, URLs and request times. `image-config.json`
is the public ARM64 image configuration. The two Hub staging manifests record all
17 target files (16,081,520,999 bytes) and all five draft files (3,849,113,002 bytes),
with byte counts/SHA-256 and, where applicable, matching Hub LFS SHA-256.

`launch-failed-distribution.json` preserves the real SparkRun local-fast-path
failure: the draft was incorrectly requested at the target revision. The final
recipe avoids that path for the draft using a pinned, checksummed pre-launch
stager. No installed SparkRun code was patched.

`preliminary-loopback/` contains the earlier loopback process's exploratory
receipts and an earlier LAN log capture. They are historical diagnostics only,
not substituted for final-process evidence and not verified by the final
process-binding gate. Their capture script versions were not separately frozen.

## Recompute

Use Python with PyYAML, e.g. the installed SparkRun virtual environment:

```bash
python scripts/qwen38/verify_deployment.py \
  evidence/qwen38-tensorfold-20261019 --negative-controls
```

The verifier requires exact runtime recipe bytes, canonical complete request
bodies, HTTP status/model identity, all required artifacts, the deterministic
fixture, fixed endpoint routes, launch/job/mod/process bindings and log coverage.
It rejects four refreshed-manifest semantic mutations plus six deployment
mutations (privilege, PID identity, job override, required artifact deletion,
kernel allocation failure and invalid event timing). The manifest is not an
independent trust root: the independently reviewed Git tree anchors the manifest,
harness, raw receipts and canonical verifier constants together.

Capture commands are implemented in `scripts/qwen38/`:
`launch.py`, `accept.py`, `proxy_accept.py`, `capture_runtime.py`,
`capture_logs.py`, `static_check.py`, `freeze.py`, and the two verifiers. Host-
local paths and IPs in these deployment receipts are intentionally preserved.
The semantic image fixture is generated deterministically without external media.

## Not established

No claim of full-262K retrieval, YaRN/1M context, video, 50 simultaneous images,
upstream performance parity, benchmark TPS, broad model quality, sustained uptime,
automatic engine restart or host-reboot recovery. The API is unauthenticated and
root-user, with host networking/IPC, IPC_LOCK, GPU access and no RDMA mount; Docker
privileged is false. Default recipe binding is loopback; this accepted process
uses the documented explicit private-LAN override. Keep earlyoom inactive while
loaded and restore it only after unloading and checking safe headroom. Review the
license/provenance discrepancy and NVIDIA terms in the deployment documentation.
