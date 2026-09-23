# Historical GLM-5.3 f906ee9 regression inputs

`recipe-mod.tar.gz` contains the exact recipe and complete mod tree from
sparkrun-recipes commit `4f8ecea3a513a151d772de1e067ebcbd72b2f684`:

- `recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml`
- `mods/glm-5.3-flash-exl3-upstream-850k/` (including `SHA256SUMS` and upstream license)

The upstream source revision is `f906ee990596486e10ddbe381efa6f0e496f77e3`.
The archive contains 123 regular files, preserving their original bytes and
modes. Its SHA-256 is
`8243a2b4caae1f8a6621a4fc7355379db7832e710f6c4b871cd30964ad0a0b2f`.
It was produced from the baseline rollback archive, not the refreshed working
tree. Archive member order is lexical and gzip timestamp/name are normalized.

`tests/test_glm53_latest_recipe.py` retains its filename for existing test
commands but now explicitly tests historical contracts. A module-scoped pytest
fixture authenticates this archive before extracting it into pytest's temporary
directory. Tests bind the recipe hash and complete mod-manifest hash to the
original `evidence/glm53-exl3-850k-20260913/launch-receipt.json`, validate every
mod file, and execute the unchanged historical verifier with explicit
`--recipe` and `--mod-dir` paths. Neither git history nor network access is
needed to load the fixture; shallow clones and source exports work unchanged.
Compression avoids duplicate vendored pytest collection and keeps the fixture
compact while preserving full manifest coverage and the original AGPL-3.0
upstream license. The archive does not contain model checkpoints.

Historical evidence stays in its original evidence directory; no receipts,
run IDs, timestamps, bindings, or verdicts are rewritten. A CPU regression pass
is **not** fresh live validation or publication approval for the active recipe.
Active refresh coverage belongs to `tests/test_glm53_refresh_20260923.py` and
`tests/test_glm53_refresh_evidence.py`. Repository-wide credential-scan and
pytest-configuration checks in the historical module still inspect the current
repository; they are not frozen to this fixture.

Run from the repository root:

```sh
uv run --with pytest --with pyyaml --with detect-secrets python -m pytest tests/test_glm53_latest_recipe.py -q
```

Do not regenerate this fixture during a recipe refresh. A different deployment
requires separate evidence and tests, not rebinding these historical receipts.
