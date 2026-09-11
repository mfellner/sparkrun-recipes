# Detect-secrets release gate

Run the reproducible all-files gate from the repository root:

```bash
python3 scripts/verify_detect_secrets.py --root .
```

The committed `.secrets.baseline` was generated with detect-secrets 1.5.0 using:

```bash
detect-secrets scan --all-files --no-verify \
  --exclude-files '(^|/)(\.secrets\.baseline$|\.secrets\.occurrences\.json$|\.pytest_cache/|__pycache__/|\.ruff_cache/|\.mypy_cache/)' . > .secrets.baseline
```

Every baseline entry is explicitly adjudicated with `is_secret: false`. The reviewed candidates fall into these repository-specific categories:

- immutable commit IDs, checksums, image digests, request hashes, and run identifiers detected by entropy plugins;
- committed binary/data fixtures and deliberately encoded synthetic test payloads;
- synthetic sanitizer-negative-control strings;
- the pinned upstream local-only dummy API value and API environment-variable name.

The verifier copies the reviewed baseline and rescans every file with the baseline's exact plugin/filter configuration. `.secrets.occurrences.json` also binds the cardinality and digest of raw per-file scanner occurrences, so duplicating an already-reviewed value cannot collapse into an existing baseline identity. The gate fails if the command errors, writes stdout or stderr, introduces or removes a finding, changes occurrence cardinality, produces malformed output, or leaves any finding unadjudicated. The release claim is **zero unadjudicated findings**, not zero raw entropy candidates.

Regenerate and re-review the baseline after any legitimate candidate-set change. Never mark a candidate false merely to make the gate pass.
