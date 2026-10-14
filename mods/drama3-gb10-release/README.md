# Minimal Drama3 compact-head wrapper

Build from repository root using this Dockerfile. The immutable model-free ARM64
base preserves original Fish weights/codec/API/reference/security behavior.
Only compact BF16 semantic/EOS projection is admitted, with optional head-off
mode for matched controls. No new precision reduction or fine-grained streaming.

Fish source/checkpoint and default voice provenance retain their own terms; see
NOTICE and docs/drama3-performance.md. The root release manifest covers the exact
copied runtime sources, installer, patch, pins and license notices. Docker uses
explicit COPY lists; generated Python caches are not admitted build inputs.

The head initializes retained candidate weights once before graphs, with global
IDs/EOS/RAS preserved. Source weights and retained codebook weights are validated.
Hidden/candidate/codebook/scaled sampler and next hidden finite checks remain
fail-closed. Uncomputed excluded vocabulary rows cannot retain the original
full-vocabulary arithmetic-overflow rejection policy; this is explicitly not a
universal arithmetic/byte-identity claim. Real qualification scope is the
published corpus and immutable image, not presence of these sources.
