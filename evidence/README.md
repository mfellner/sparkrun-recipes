# Evidence

This directory contains executable validation tools and preserved receipts for recipe releases.

For the 2026-09-03 Qwen3.8 Flash Next addition and GLM-5.3 Flash memory-setting adjustment, start with:

- `qwen3.8-flash-next-nvfp4-20260902/README.md`
- `glm53-pair-20260903/README.md`
- `final-secure-audit-20260903.json`

Acceptance claims are limited to the exact artifacts, topology, model revisions, image identities, commands, and endpoints named in those receipts. Failed and superseded runs remain labeled as such and do not count as passes.

## Required post-publish round-trip

After merging, verify the published artifact rather than trusting the working tree:

1. Fetch `origin`, read the merge commit from `origin/main`, and verify GitHub's `refs/heads/main` resolves to the same commit.
2. Download both changed recipe files from `https://raw.githubusercontent.com/mfellner/sparkrun-recipes/$MERGE_SHA/...`; compare each byte-for-byte with the corresponding committed local file and assert its immutable model/image/source pins.
3. Validate both exact downloaded files with SparkRun 0.3.6.
4. Run `sparkrun registry update mfellner` on the deployment controller, then resolve both public names with:
   - `sparkrun recipe show @mfellner/qwen3.8-flash-next-nvfp4-dual-spark-1m --json`
   - `sparkrun recipe show @mfellner/glm-5.3-flash-exl3-dflash2-dual-spark-1m --json`
5. On controllers that own the validated clusters, run trusted namespaced dry-runs and local-file dry-runs against `spark-ring3` for Qwen and `spark-pair2` for GLM. Compare the rendered serve command and hook sections byte-for-byte; normalized `recipe show --json` is not a lossless substitute.
6. Check repository CI and record explicitly when no workflows are configured.

Publication is incomplete until every step passes.
