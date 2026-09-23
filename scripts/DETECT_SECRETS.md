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

## Refresh review (2026-09-23, initial tree)

The initial refresh review added 911 candidate identities and removed two superseded upstream revision identities. Each new candidate's source line was inspected, including all occurrences; no existing retained identity changed occurrence count. The reviewed additions are:

- 900 hexadecimal entropy identities: Hugging Face model commit/blob/LFS hashes, image recipe stamps, source and archive provenance, patch/golden-output checksums, and immutable recipe/test revision pins;
- 11 Base64 entropy identities that are not encoded credentials: six image environment assignments (`UV_INDEX_STRATEGY`, `UV_PYTHON_INSTALL_DIR`, `VLLM_USAGE_SOURCE`, each in two image records) and five synthetic model/path/configuration assignment strings in upstream launcher tests.

No real credential was identified among these additions. The resulting baseline contains 1,620 adjudicated false-positive identities and the occurrence manifest binds 2,269 raw occurrences. Scanner plugins, filters, exclusions, and fail-closed verifier code are unchanged. The all-files scan does not unpack compressed archives; the historical fixture archive is not evidence of credential coverage of its decompressed contents.

These counts describe the initial refresh tree, not future runtime receipts. After adding or replacing evidence, rerun the gate, inspect every changed identity and occurrence in its actual source context, preserve previous adjudications only for retained identities, and regenerate the raw occurrence manifest using `raw_occurrence_summary` from `scripts/verify_detect_secrets.py`. Do not merely copy new counts or set every new row to false.

## Final raw-receipt publication review (2026-09-23)

The final review covered all 66 non-cache files then present in
`evidence/glm53-exl3-850k-20260923/`, including every raw launch, serving,
qualification, failed-attempt, diagnostic HTTP, and operations log. Review
included nested JSON stdout/body records, effective process argv/environment,
credential-related fields and source contexts, and all new scanner occurrences.
All 126 recorded `response_base64` payloads were decoded and checked against both
`response_text` and parsed `response`; every comparison matched. Encoded HTTP
responses contain model listings, controlled test completions, and expected API
errors, not authentication material.

Relative to the initial refresh baseline, the reviewed additions are:

| False-positive category | New identities | Raw occurrences |
| --- | ---: | ---: |
| Hex entropy: source revisions and file, request, config, launch, qualification, and receipt checksums | 635 | 957 |
| Base64 entropy: encoded HTTP response bodies | 57 | 57 |
| Base64 entropy: eight runtime cache/workspace assignments in each of two GPU qualification records | 16 | 16 |
| Base64 entropy: deliberately invalid uppercase-placement launch-ID test fixture | 1 | 1 |
| **Total additions** | **709** | **1,031** |

No identities were removed. The retained 1,620 identities still account for
exactly 2,269 raw occurrences, with the same occurrence digest as the initial
review. The final baseline contains **2,329 adjudicated false-positive
identities** and the manifest binds **3,300 raw occurrences**. Detector totals
are 2,134 Hex entropy, 178 Base64 entropy, and 17 previously reviewed Secret
Keyword identities.

The `/health` record in `operations.json` was manually reviewed even though it
adds no baseline candidate: its API-key/token/hash/alias fields contain the
literal `litellm-internal-health-check` marker or empty metadata, not a real
credential. The Hugging Face launch warnings name an unset credential variable;
they do not expose a token. No genuine credential or credential-publication
blocker was identified in the reviewed additions.

These are **raw, audited records, not sanitized records**. LAN addresses, local
user paths, process/container identifiers, and operational details remain
visible. This review does not expand the archive-coverage claim above or imply
that future appended logs are approved. Plugins, filters, exclusions, and the
fail-closed verifier are unchanged.

Reproduce the gate with the pinned scanner through uv:

```bash
uv run --with detect-secrets==1.5.0 python scripts/verify_detect_secrets.py --root .
```

The final gate must report `passed: true`, 2,329 findings/adjudicated false
positives, 3,300 raw occurrences, empty stderr, and no errors. Any subsequent
candidate or occurrence change requires renewed source-context review.
