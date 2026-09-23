# Test the cooperative MoE extension on two Sparks

This is the complete manual opt-in procedure for an **already working
GLM-5.3-Flash TP2 installation**. Use a dedicated Bash shell on the head Spark,
with this tree checked out, the checkpoint already installed, and passwordless
SSH/Docker access to the configured worker. The measured configuration is
DFlash2 k=7, `MAX_NUM_SEQS=4`, E3 grouped prefill left on.

**Activation is a selected overlay, not a standalone environment toggle.**
`GLM53_COOPERATIVE_MOE=1 ./start.sh start` by itself does not import the adapter
or deploy its binary. The sequence below builds and verifies artifacts, copies
them to both nodes, then selects the generated overlay in `.env`.

Do **not** copy or load the DS4.1 `cooperative_moe.so`. That kernel is locked to
DeepSeek-V4.1 and will not match these shapes. Do not stop a live DS4.1 serve
for this procedure.

Read the rollback section before starting. Steps 2–7 require a maintenance
window; the stop command cancels any remaining GLM-5.3-Flash requests. A failed
check is a reason to roll back, not to disable the integrity checks.

## 1. Read the existing configuration and save a rollback copy

Run from the recipe root. Only source your own trusted `.env`; do not print or
commit its contents.

```bash
set -euo pipefail
COOP_REPO="$(pwd -P)"
test -f "$COOP_REPO/start.sh"
test -f "$COOP_REPO/.env"
test -f "$COOP_REPO/extensions/cooperative_moe/build.sh"
source "$COOP_REPO/.env"

COOP_WORKER_USER="${WORKER_USER:-$USER}"
COOP_WORKER="${WORKER_SSH:-${COOP_WORKER_USER}@${WORKER_IP:-10.0.0.2}}"
if [ "$COOP_WORKER_USER" = "$USER" ]; then
  COOP_WORKER_HOME="${WORKER_HOME:-$HOME}"
else
  COOP_WORKER_HOME="${WORKER_HOME:-/home/${COOP_WORKER_USER}}"
fi
COOP_HEAD_CACHE="${CACHE_ROOT:-$HOME/.cache/vllm-glm53-flash}"
COOP_WORKER_CACHE="${WORKER_VLLM_CACHE:-$COOP_WORKER_HOME/.cache/vllm-glm53-flash}"
COOP_HEAD_CONTAINER="${CONTAINER_HEAD:-glm53-exl3-head}"
COOP_WORKER_CONTAINER="${CONTAINER_WORKER:-glm53-exl3-worker}"
COOP_PORT="${PORT:-8888}"

mkdir -p "$COOP_REPO/logs"
COOP_RUN="$(mktemp -d "$COOP_REPO/logs/cooperative-moe.XXXXXX")"
COOP_SLOT="${COOP_RUN##*/}"
COOP_HEAD_STAGE="$COOP_HEAD_CACHE/$COOP_SLOT"
COOP_WORKER_STAGE="$COOP_WORKER_CACHE/$COOP_SLOT"
COOP_RUNTIME="/root/.cache/vllm/$COOP_SLOT"
install -m 600 "$COOP_REPO/.env" "$COOP_RUN/original.env"
printf 'Rollback configuration: %s\n' "$COOP_RUN/original.env"
```

## 2. Drain requests, stop GLM-5.3-Flash, and pin the build image

Stop clients before continuing. Use the **same image you already serve**, then
record its ID on both nodes. A reference for the default GHCR `:exl3` tag at
the time this package was built:

`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:eecb36e14dc34c92d46827fde7b09f7e0bf27e27c426ece126376c02dea6cd2f`
→ local Id `sha256:9581c4c7425786be27a7904a78c01bb27ae138e33840e605ecc706e76c761900`.

```bash
./start.sh stop
COOP_IMAGE="${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3}"
docker image inspect --format '{{.Id}}' "$COOP_IMAGE"
ssh -o BatchMode=yes "$COOP_WORKER" "docker image inspect --format '{{.Id}}' '$COOP_IMAGE'"
```

Both IDs must match. Do not substitute a mutable `latest` tag and assume it
reproduces the validated artifact.

## 3. Build in the pinned image

The recipe image has nvcc but not git. Archive the upstream pin on the host,
then compile in the container with `/work` as the output mount. No GPU is
assigned to the build.

```bash
git clone --filter=blob:none --no-checkout https://github.com/turboderp-org/exllamav3.git "$COOP_RUN/exllamav3"
git -C "$COOP_RUN/exllamav3" checkout --detach 02aef45cd681b960a00afcd0749a4ab99e6c1bfe
mkdir "$COOP_RUN/staged" "$COOP_RUN/build"
git -C "$COOP_RUN/exllamav3" archive 02aef45cd681b960a00afcd0749a4ab99e6c1bfe exllamav3/exllamav3_ext |
  tar -x -C "$COOP_RUN/staged"
printf '%s\n' 02aef45cd681b960a00afcd0749a4ab99e6c1bfe > "$COOP_RUN/staged/.glm53_coop_upstream_pin"
docker run --rm --network none --cpus 4 --memory 8g --memory-swap 8g \
  --user "$(id -u):$(id -g)" \
  -v "$COOP_REPO/extensions/cooperative_moe:/src:ro" \
  -v "$COOP_RUN/staged:/upstream:ro" \
  -v "$COOP_RUN/build:/work" \
  --entrypoint bash "$COOP_IMAGE" /src/build.sh /upstream /work
```

The expected `cooperative_moe.so` digest is
`aa3fe5e9387c7e0d42d685fb2ca8a5fb959ad956600236baac078a9076c17a1c`.
`prepare_profile.py` enforces this and the stock/runtime source digests. If a
clean rebuild differs, preserve the build log and stop.

## 4. Generate the overlay and deploy identical artifacts to both nodes

The launcher copies the selected overlay to the worker, but **does not copy the
native library or adapter**. Copy them explicitly. The launcher mounts each
host's vLLM cache at `/root/.cache/vllm`.

```bash
mkdir -p "$COOP_HEAD_STAGE"
install -m 644 "$COOP_RUN/build/cooperative_moe.so" \
  "$COOP_REPO/extensions/cooperative_moe/runtime.py" "$COOP_HEAD_STAGE/"
python3 "$COOP_REPO/extensions/cooperative_moe/prepare_profile.py" \
  --stock "$COOP_REPO/overlay/exl3.py" \
  --artifacts "$COOP_HEAD_STAGE" \
  --runtime-directory "$COOP_RUNTIME" \
  --output "$COOP_HEAD_STAGE/exl3-cooperative.py"
install -m 644 "$COOP_REPO/extensions/cooperative_moe/test_cuda_integration.py" \
  "$COOP_REPO/tests/test_exl3_overlay.py" "$COOP_HEAD_STAGE/"

ssh -o BatchMode=yes "$COOP_WORKER" "mkdir -p '$COOP_WORKER_STAGE'"
scp "$COOP_HEAD_STAGE/cooperative_moe.so" "$COOP_HEAD_STAGE/runtime.py" \
  "$COOP_HEAD_STAGE/exl3-cooperative.py" "$COOP_HEAD_STAGE/test_cuda_integration.py" \
  "$COOP_HEAD_STAGE/test_exl3_overlay.py" "$COOP_WORKER:$COOP_WORKER_STAGE/"
(cd "$COOP_HEAD_STAGE" && sha256sum cooperative_moe.so runtime.py exl3-cooperative.py \
  test_cuda_integration.py test_exl3_overlay.py) > "$COOP_RUN/SHA256SUMS"
scp "$COOP_RUN/SHA256SUMS" "$COOP_WORKER:$COOP_WORKER_STAGE/SHA256SUMS"
ssh -o BatchMode=yes "$COOP_WORKER" "cd '$COOP_WORKER_STAGE' && sha256sum -c SHA256SUMS"
```

## 5. Run the packaged GPU gate on each node

GLM-5.3-Flash and other GPU workloads must still be stopped. These containers
do not load the full checkpoint. Require a final `status: pass` record and a
zero process exit code on each node. A pass is not bitwise equality to stock.

```bash
docker run --rm --network none --gpus all --cpus 2 --memory 8g --memory-swap 8g \
  -e GLM53_COOP_MAINTENANCE_TEST=1 -e MAX_JOBS=2 -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$COOP_HEAD_CACHE:/root/.cache/vllm" \
  -v "$COOP_HEAD_STAGE/exl3-cooperative.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro" \
  -v "$COOP_HEAD_STAGE/test_exl3_overlay.py:/opt/glm53/test_exl3_overlay.py:ro" \
  -v "$COOP_HEAD_STAGE/test_cuda_integration.py:/opt/glm53/test_cuda_integration.py:ro" \
  --entrypoint python3 "$COOP_IMAGE" /opt/glm53/test_cuda_integration.py \
  2>&1 | tee "$COOP_RUN/cuda-head.log"

ssh -o BatchMode=yes "$COOP_WORKER" "docker run --rm --network none --gpus all --cpus 2 --memory 8g --memory-swap 8g \
  -e GLM53_COOP_MAINTENANCE_TEST=1 -e MAX_JOBS=2 -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v '$COOP_WORKER_CACHE:/root/.cache/vllm' \
  -v '$COOP_WORKER_STAGE/exl3-cooperative.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro' \
  -v '$COOP_WORKER_STAGE/test_exl3_overlay.py:/opt/glm53/test_exl3_overlay.py:ro' \
  -v '$COOP_WORKER_STAGE/test_cuda_integration.py:/opt/glm53/test_cuda_integration.py:ro' \
  --entrypoint python3 '$COOP_IMAGE' /opt/glm53/test_cuda_integration.py" \
  2>&1 | tee "$COOP_RUN/cuda-worker.log"
```

## 6. Select the profile in `.env` and start both nodes

Print the generated overlay path:

```bash
printf 'EXL3_OVERLAY_HOST=%s\n' "$COOP_HEAD_STAGE/exl3-cooperative.py"
```

Edit `.env`: **replace** any existing `EXL3_OVERLAY_HOST` rather than leaving
conflicting entries. Keep E3 and the rest of the working topology. Do not
blindly copy another cluster's `.env`.

```dotenv
EXL3_OVERLAY_HOST=/absolute/path/printed/above/exl3-cooperative.py
EXL3_FUSED_MOE=1
EXL3_FAT_GROUPED=1
EXL3_FAT_KERNEL=0
EXL3_TEMP_ROWS_FUSED=32
SPEC_METHOD=dflash
DFLASH_TOKENS=7
MAX_NUM_SEQS=4
```

`EXL3_OVERLAY_HOST` must be set in `.env` when an older override already exists
there: the launcher sources `.env` after reading the process environment.
The generated overlay calls `install(..., enabled=True)`. `start.sh` accepts
this footer: the stock `Exl3Config` closer must still be present in the body so
a truncated copy cannot hide behind the install lines.

```bash
IMAGE="$COOP_IMAGE" SPEC_METHOD=dflash DFLASH_TOKENS=7 \
  MAX_NUM_SEQS=4 EXL3_FUSED_MOE=1 EXL3_FAT_GROUPED=1 EXL3_FAT_KERNEL=0 \
  EXL3_TEMP_ROWS_FUSED=32 \
  SKIP_PULL=1 SKIP_BUILD=1 SKIP_SHIP=1 SKIP_SYNC=1 \
  ./start.sh start 2>&1 | tee "$COOP_RUN/start.log"
```

`SKIP_SYNC=1` assumes the pre-existing weight setup still works. For later
starts, keep the verified image reference in `.env` as `IMAGE` if this profile
remains selected.

## 7. Verify activation, then send a short request

Wait for warmups, not just the first HTTP health response. Check both nodes for
the explicit activation message.

```bash
sha256sum "$COOP_HEAD_STAGE/exl3-cooperative.py"
docker logs "$COOP_HEAD_CONTAINER" 2>&1 | grep -F 'Fixed-shape cooperative MoE enabled'
ssh -o BatchMode=yes "$COOP_WORKER" "docker logs '$COOP_WORKER_CONTAINER' 2>&1 | grep -F 'Fixed-shape cooperative MoE enabled'"
docker exec "$COOP_HEAD_CONTAINER" sha256sum /opt/glm53/exl3.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
curl --max-time 10 -fsS "http://127.0.0.1:$COOP_PORT/health"
```

Then a short chat completion against `GLM-5.3-Flash-EXL3` (add the configured
API key header if `VLLM_API_KEY` is set). Only proceed to sparkDash prose ×1/×2
and structured ×1 if the gates and startup pass. Compare against the current
README: prose 32.1 / 41.2 agg, structured ×1 62.9 tok/s. Keep E3 on.

## 8. Roll back

Drain clients again. Restore the complete saved configuration:

```bash
./start.sh stop
install -m 600 "$COOP_RUN/original.env" "$COOP_REPO/.env"
env -u IMAGE -u SPEC_METHOD -u DFLASH_TOKENS -u MAX_MODEL_LEN -u MAX_NUM_SEQS \
  -u MAX_NUM_BATCHED_TOKENS -u LANGUAGE_MODEL_ONLY -u EXL3_FUSED_MOE \
  -u EXL3_TEMP_ROWS_FUSED -u EXL3_FAT_KERNEL -u EXL3_FAT_GROUPED \
  -u EXL3_OVERLAY_HOST \
  ./start.sh start
```

This returns to the previous configuration, which is stock only if you started
from stock. Versioned artifacts can remain for inspection; their presence does
not enable the feature. No model weights, caches or rollback files need
deletion.

## Validation status of this procedure

Host-side tests, the pinned native rebuild, launcher overlay wiring, and
command syntax have been checked. The packaged GPU gate passed on this head
(18 cases, peak-normalized vs stock fused; E3 still resolved grouped; rows 40
stayed stock). A two-node opt-in start **has** been executed; live geometry-1
decode numbers and rollback are in
[cooperative-moe-handoff.md](cooperative-moe-handoff.md). Worker-node GPU gate
and sparkDash prose ×2 are still open.
