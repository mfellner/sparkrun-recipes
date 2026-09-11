# GLM-5.3 Flash EXL3 850K live validation

Evidence state: **LIVE_CAPTURE_PASSED; PUBLICATION_PENDING**.

This package binds the reviewed 850K recipe to one deterministic SparkRun
launch, a fresh direct/proxy acceptance run, synchronized load telemetry, and a
dual-rank runtime capture. It does not treat serving health alone as release
approval.

- SparkRun workload: `sparkrun_3d13e8eba3fa512a_38acb2ac0fc5`
- Acceptance run ID: `1941bd91758d28de`
- Acceptance interval: `2026-09-10T15:06:13.227021+02:00` through `2026-09-10T15:07:39.580214+02:00`
- Runtime capture completed: `2026-09-11T04:20:42.873080+02:00`
- Mia source: `1caea9a10b26ae93b88d08e82d1e7abb0dc45a42`
- Image digest: `sha256:eecb36e14dc34c92d46827fde7b09f7e0bf27e27c426ece126376c02dea6cd2f`
- Main model: `024db9f7e9871e8efdf21538ba55af7442be3cd5`
- DFlash2: `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`
- Configured context: 850,000 tokens
- Largest completed semantic request: 110,035 prompt tokens
- Acceptance checks: 18 of 18 passed
- Secret scan: 627 reviewed false-positive identities covering 1,249 raw occurrences, with zero unadjudicated findings;
  reproduce with `python3 scripts/verify_detect_secrets.py --root .`

## What the live receipts prove

`launch-receipt.json` was recorded before Docker creation. It binds the current
recipe SHA-256, exact reviewed YAML command, external mod-manifest digest,
deterministic container name, launch epoch, and launch argv. `runtime.json`
then binds SparkRun's raw recipe command and both rank-specific wrapper/runtime
argv back to that receipt.

Both hosts expose a closed Docker process inventory. The verifier requires the
actual lifecycle rather than an invented shell chain:

- the container launcher, `serve_wrapper.sh`, and watchdog share the external
  containerd shim parent;
- the launcher, container shell, keepalive, watchdog, resource tracker,
  EngineCore, and rank workers have exact PPID/PGID/SID relationships;
- vLLM is the wrapper's direct child and its own process-group/session leader;
- host and container vLLM PIDs join through `/proc/<host-pid>/status` `NSpid`;
- the wrapper's external parent is the exact containerd shim for the inspected
  64-hex container ID;
- NCCL ownership joins the vLLM container PID;
- rank 0 owns the sole container-namespace `0.0.0.0:8000` listener; and
- rank 1 has an explicit host-namespace receipt with `listeners: []`.

The structured serve-marker receipt contains only bounded facts from the live
serve log: all required E3/grouped-kernel, FP8 KV, indexer-workspace, CUDA-graph,
and DFlash2 markers are present; the exact E3 execution counters are positive;
and the selected fatal-signature list is empty. Full serve logs may be redacted
rather than committed when they contain unsafe command-like content.

The acceptance run passed model discovery, exact completions, C4 concurrency,
110,035-token retrieval, synchronized load, reasoning controls, tool calling,
two deterministic image/OCR fixtures, remote-media rejection, and both direct
and proxy routes. The command's explicit video limit is tested with
`video-tiny.gif`, a deterministic two-frame 224x224 fixture:

- MIME type: `image/gif`
- Size: 835 bytes
- SHA-256: `b89ebcdbedac896cc0ee9645f92000a1780d2095cfe76aa7a0b2cd38ef93b5f0`
- Direct route: exact HTTP 400 rejection envelope
- Proxy route: exact HTTP 400 rejection envelope

## Reproduction and binding

A replacement acceptance must use a new 16-hex identifier with
`acceptance.py --run-id <RUN_ID>`. A replacement runtime capture must use the
same identifier with `capture_runtime.py --acceptance-run-id <RUN_ID>` and the
immutable launch epoch/receipt. `verify.py` pins that identifier in
`EXPECTED_ACCEPTANCE_RUN_ID`. Receipts must be generated live; listener,
process, video, telemetry, and proxy results must never be inferred or
synthesized.

The recipe keeps SparkRun defaults only for estimation, status, labels, and
proxy discovery. Its executable command contains literal values and does not
interpolate caller-overridable fields. The reviewed YAML is the sole command
trust root; `command_contract.py` mechanically derives both rank commands.

Remote media is restricted to the non-resolving `media.invalid` sentinel;
inline `data:` images remain supported. The API is unauthenticated, root-run,
and host-networked. Inference, distributed-runtime, NCCL, and proxy port `4000`
are LAN-only untrusted-client boundaries and must remain behind a trusted
firewall. Keep `earlyoom` inactive while the model remains loaded.

The complete vendored Mia tree is archival provenance.
`upstream/scripts/boot_candidate.sh` is not invoked by this recipe. Runtime
behavior is limited to the checksum-pinned paths called by `run.sh`,
`serve_wrapper.sh`, and `postready_gate.sh`. The externally pinned local
mod-manifest digest must match both independently captured rank manifests and
complete `sha256sum -c` results.

## Remaining release gates

Exact artifact hashes, `SHA256SUMS`, and `static-validation.log` must match one
immutable candidate. That candidate must pass the canonical verifier, every
adversarial negative control, repository pytest, compatibility/source-parity
and lifecycle suites, recipe validation, trusted dry-run, diff checks, and the
credential/private-key scan.

Publication then requires two independent fail-closed approvals and explicit
owner approval of the identical staged-tree hash. Any byte changed after
staging invalidates the tests and both approvals. Commit and push are followed
by exact-commit GitHub raw-byte, registry-resolution, trusted namespaced
dry-run parity, zero-workflow, live-health, and clean-tree checks.
