# DeepSeek V4 Flash 0731 DSpark upstream hotfix mod

This SparkRun mod vendors the exact default runtime hotfix set from:

- Repository: `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
- Commit: `a462a9e541c684b58c7f380bbd92c7d851557f31`
- Base image expected by the recipe: `ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
- Model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`

`run.sh` executes inside every rank container before vLLM starts. It verifies `SHA256SUMS`, verifies the pinned model encoder checksum, applies the upstream patches in upstream order, and runs each available status check fail-closed.

The assistant-final continuation and GPU `thinking_token_budget` patches are included for byte-level upstream completeness but remain disabled by default, matching upstream. The three upstream opt-out gates for Issue 22, spin-wait, and the seven-patch performance loop are preserved and default off. API-key modes are not implemented by this recipe; the redaction patch is included only as uninvoked upstream source provenance. This recipe is unauthenticated and requires a trusted private network.

Do not edit vendored `hotfix-*` files locally. Refresh them only by copying from a newly audited exact upstream commit, regenerate `SHA256SUMS`, and repeat the upstream CPU gates, disposable-container application on every rank, live launch, and behavioral acceptance suite.
