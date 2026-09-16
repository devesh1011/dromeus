#!/usr/bin/env bash

set -uo pipefail

runtime_root="$1"
rank="$2"
master_addr="$3"
world_size="$4"
seed="$5"
profile="$6"
base_root="/home/ubuntu/noloco-nccl-w16-s29-r2-20260914"
image="150911080841.dkr.ecr.us-east-1.amazonaws.com/dromeus-m2-benchmark@sha256:1b40d3774dc864f1f4450720f91243f2e379de8ec7d12805d120ec1ef1419a3c"

mkdir -p "$runtime_root"
exec 9>"$runtime_root/launch.lock"
flock -n 9 || { echo "Duplicate launch rejected: lock held" >&2; exit 73; }
if [ -e "$runtime_root/started-at.txt" ]; then
  echo "Duplicate launch rejected: immutable attempt already started" >&2
  exit 73
fi
mkdir -p "$runtime_root/final-checkpoints" "$runtime_root/reference-output" "$base_root/cifar-cache"
(set -o noclobber; date -u +%Y-%m-%dT%H:%M:%SZ >"$runtime_root/started-at.txt") || exit 73

sudo -n docker run --rm --gpus all --network host --shm-size 1g \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH=/workspace/dromeus \
  --env MASTER_ADDR="$master_addr" \
  --env MASTER_PORT=29500 \
  --env NCCL_SOCKET_IFNAME=ens5 \
  --env NCCL_SOCKET_FAMILY=AF_INET \
  --env NCCL_IB_DISABLE=1 \
  --env NCCL_PORT_RANGE=50000-50099 \
  --env NCCL_DEBUG=INFO \
  --env NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH \
  --env NCCL_ASYNC_ERROR_HANDLING=1 \
  --env NCCL_BLOCKING_WAIT=1 \
  --env NCCL_TIMEOUT=1800 \
  --env NCCL_FINAL_CHECKPOINT_DIR="$runtime_root/final-checkpoints" \
  --volume "$base_root:$base_root" \
  --workdir /workspace/dromeus \
  "$image" \
  .venv/bin/torchrun \
    --nnodes="$world_size" \
    --nproc-per-node=1 \
    --node-rank="$rank" \
    --master-addr="$master_addr" \
    --master-port=29500 \
    "$base_root/wrapper/capture_final.py" \
    --experiment "$base_root/frozen/experiment.yaml" \
    --profile "$profile" \
    --world-size "$world_size" \
    --seed "$seed" \
    --backend nccl \
    --dataset-cache "$base_root/cifar-cache" \
    --output-dir "$runtime_root/reference-output"
rc=$?
(set -o noclobber; printf '%s\n' "$rc" >"$runtime_root/exit-code.txt")
(set -o noclobber; date -u +%Y-%m-%dT%H:%M:%SZ >"$runtime_root/ended-at.txt")
exit "$rc"
