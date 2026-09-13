# Evidence

This directory contains executable validation tools and preserved receipts for recipe releases.

For the 2026-09-03 Qwen3.8 Flash Next addition and GLM-5.3 Flash memory-setting adjustment, start with:

- `qwen3.8-flash-next-nvfp4-20260902/README.md`
- `glm53-pair-20260903/README.md`
- `final-secure-audit-20260903.json`

Acceptance claims are limited to the exact artifacts, topology, model revisions, image identities, commands, and endpoints named in those receipts. Failed and superseded runs remain labeled as such and do not count as passes.

## GLM-5.3 EXL3 850K refresh

`glm53-exl3-850k-20260913/` is **LIVE_CAPTURE_PASSED** for deterministic workload `sparkrun_f906ee990596486e_20260913c411` and acceptance run `f906c41120260913`. Its exact command chain, direct/proxy functional matrix, synchronized load telemetry, process/PID-namespace joins, external shim lineage, rank-0 positive and rank-1 negative listener receipts, latest hybrid-DFlash/per-group-APC source state, NCCL/RDMA identity, and deterministic direct/proxy video-zero rejections are bound by the canonical verifier and adversarial controls.

Publication remains pending until the exact staged tree passes all local gates, receives two independent fail-closed approvals and explicit owner approval, and is committed and pushed. Post-publication verification remains pending until that approved exact commit is checked through GitHub raw bytes and registry resolution.

## Required post-publish round-trip

GitHub and registry results are **PENDING until an approved push**. After the exact reviewed commit is pushed, set `PUBLISH_SHA` to that immutable 40-hex commit (never to a moving branch name) and run:

```bash
set -euo pipefail
RECIPE=recipes/glm-5.3-flash-exl3-dflash2-dual-spark-850k.yaml
MOD_TREE=mods/glm-5.3-flash-exl3-upstream-850k
NAME=@mfellner/glm-5.3-flash-exl3-dflash2-dual-spark-850k
: "${PUBLISH_SHA:?set the approved 40-hex published commit}"
APPROVED_TREE="${APPROVED_TREE:?set the approved 40-hex staged tree}"
[[ "$PUBLISH_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$APPROVED_TREE" =~ ^[0-9a-f]{40}$ ]]
CONTAINER_NAME="sparkrun_postpublish_glm53_850k"
git fetch origin main
test "$(git rev-parse origin/main)" = "$PUBLISH_SHA"
test "$(git ls-remote origin refs/heads/main | cut -f1)" = "$PUBLISH_SHA"
test "$(git rev-parse "$PUBLISH_SHA^{tree}")" = "$APPROVED_TREE"
PUBLISHED_TREE="$(mktemp -d)"
trap 'rm -rf "$PUBLISHED_TREE"' EXIT
git archive "$PUBLISH_SHA" -- "$RECIPE" "$MOD_TREE" | tar -x -C "$PUBLISHED_TREE"
test -f "$PUBLISHED_TREE/$RECIPE"
test -d "$PUBLISHED_TREE/$MOD_TREE"
curl -fsS "https://raw.githubusercontent.com/mfellner/sparkrun-recipes/$PUBLISH_SHA/$RECIPE" \
  -o "$PUBLISHED_TREE/published-recipe.yaml"
cmp --silent "$PUBLISHED_TREE/$RECIPE" "$PUBLISHED_TREE/published-recipe.yaml"
sparkrun recipe validate "$PUBLISHED_TREE/$RECIPE" --json
sparkrun registry update mfellner
sparkrun recipe show "$NAME" --json > "$PUBLISHED_TREE/registry-show.json"
# Trust only after the exact-commit source audit above.
sparkrun registry trust mfellner
sparkrun run "$PUBLISHED_TREE/$RECIPE" --cluster vacation-pair2 --dry-run --trust \
  --container-name "$CONTAINER_NAME" \
  > "$PUBLISHED_TREE/local.dry-run"
sparkrun run "$NAME" --cluster vacation-pair2 --dry-run --trust \
  --container-name "$CONTAINER_NAME" \
  > "$PUBLISHED_TREE/namespaced.dry-run"
cmp --silent "$PUBLISHED_TREE/local.dry-run" "$PUBLISHED_TREE/namespaced.dry-run"
curl -fsS http://127.0.0.1:8000/v1/models > "$PUBLISHED_TREE/direct-models.json"
curl -fsS http://127.0.0.1:4000/v1/models > "$PUBLISHED_TREE/proxy-models.json"
test -z "$(git status --porcelain)"
```

The final `cmp` is the trusted namespaced dry-run versus the reviewed local-file dry-run; it binds rendered commands and hooks rather than treating normalized `recipe show --json` as lossless YAML.

The endpoint checks verify the already accepted workload without restarting it.
Any destructive namespaced relaunch requires a separate operator decision and is
not part of this publication round trip.

There are currently no repository CI workflows configured. Recheck the exact published commit with `git ls-tree -r --name-only "$PUBLISH_SHA" -- .github/workflows`; if it remains empty, record that no-CI result rather than claiming a CI pass. Publication remains incomplete until the GitHub, registry, deterministic dry-run, live-health, and clean-tree checks pass.
