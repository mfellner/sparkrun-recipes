#!/usr/bin/env bash
# SparkRun 0.3.6 native-cluster adapter for TensorFold 0.6.0.
# SparkRun owns containers, rank placement and the init network; TensorFold
# owns the actual distributed model process and OpenAI-compatible API.
set -euo pipefail

[[ $# -ge 1 ]] || { echo 'missing model snapshot' >&2; exit 2; }
model=$1; shift
rank='' nnodes='' master='' master_port='' headless=0
while (($#)); do
  case "$1" in
    --nnodes|--node-rank|--master-addr|--master-port|--served-model-name)
      (($# >= 2)) || { echo "missing value for $1" >&2; exit 2; }
      case "$1" in
        --nnodes) [[ -z "$nnodes" ]] || exit 2; nnodes=$2 ;;
        --node-rank) [[ -z "$rank" ]] || exit 2; rank=$2 ;;
        --master-addr) [[ -z "$master" ]] || exit 2; master=$2 ;;
        --master-port) [[ -z "$master_port" ]] || exit 2; master_port=$2 ;;
        --served-model-name) [[ "$2" == "$TF_SERVED_NAME" ]] || { echo 'served alias mismatch' >&2; exit 2; } ;;
      esac
      shift 2 ;;
    --headless) (( headless == 0 )) || exit 2; headless=1; shift ;;
    *) echo "unexpected SparkRun argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$nnodes" == 2 && "$rank" =~ ^[01]$ && "$master" =~ ^[0-9.]+$ && "$master_port" =~ ^[0-9]+$ ]] || {
  echo "expected exactly two native ranks and an IPv4 rendezvous: nnodes=$nnodes rank=$rank master=$master port=$master_port" >&2; exit 2;
}
[[ ( "$rank" == 0 && "$headless" == 0 ) || ( "$rank" == 1 && "$headless" == 1 ) ]] || {
  echo 'headless/rank mismatch' >&2; exit 2;
}
[[ -f "$model/config.json" && -f "$TF_DRAFTER_SNAPSHOT/config.json" ]] || {
  echo 'pinned model or DFlash2 snapshot unavailable in container' >&2; exit 2;
}
[[ "${NCCL_IB_HCA:-}" != '' && "${NCCL_IB_GID_INDEX:-}" != '' ]] || {
  echo 'SparkRun must supply NCCL_IB_HCA and NCCL_IB_GID_INDEX' >&2; exit 2;
}
# SparkRun 0.3.6 prefers a reachable management address for its native init
# and injects it as --master-addr. TensorFold requires its own master to be on
# the direct CX-7 link, so use the explicit pair selected by this recipe.
node="$TF_FABRIC_MASTER"; [[ "$rank" == 1 ]] && node="$TF_FABRIC_WORKER"
netdev=$(python3 /workspace/mods/glm53-tensorfold-1m/fabric.py \
  "$rank" "$TF_FABRIC_MASTER" "$node" "$NCCL_IB_HCA" "$NCCL_IB_GID_INDEX")
export NCCL_SOCKET_IFNAME="$netdev" GLOO_SOCKET_IFNAME="$netdev"
args=(serve "$model" --tp 2 --rank "$rank" --master "$TF_FABRIC_MASTER" --master-port "$master_port"
      --drafter "$TF_DRAFTER_SNAPSHOT" --context "$TF_CONTEXT"
      --parallel "$TF_PARALLEL" --max-tokens "$TF_MAX_TOKENS")
if [[ "$rank" == 0 ]]; then
  args+=(--name "$TF_SERVED_NAME" --host "$TF_HOST" --port "$TF_PORT")
fi
[[ "$TF_THINKING" == 1 ]] && args+=(--thinking) || args+=(--no-thinking)
[[ "$TF_VISION" == 1 ]] && args+=(--vision)
[[ "$TF_VISION_URLS" == 1 ]] && args+=(--vision-urls)
printf 'TensorFold SparkRun rank=%s fabric=%s/%s sparkrun-init=%s:%s HCA=%s GID=%s\n' \
  "$rank" "$node" "$netdev" "$master" "$master_port" "$NCCL_IB_HCA" "$NCCL_IB_GID_INDEX"
exec tensorfold "${args[@]}"
