#!/usr/bin/env bash

set -uo pipefail

runtime_root="$1"
rank="$2"
master_addr="$3"
world_size="$4"
seed="$5"
base_root="/home/ubuntu/dromeus-nccl-w4-s17-20260909"
image="150911080841.dkr.ecr.us-east-1.amazonaws.com/dromeus-m2-benchmark@sha256:1b40d3774dc864f1f4450720f91243f2e379de8ec7d12805d120ec1ef1419a3c"

mkdir -p "$runtime_root/final-checkpoints" "$runtime_root/reference-output" "$runtime_root/cifar-cache"
date -u +%Y-%m-%dT%H:%M:%SZ >"$runtime_root/started-at.txt"

sudo docker run --rm --gpus all --network host --shm-size 1g \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH=/workspace/dromeus \
  --env NCCL_SOCKET_IFNAME=ens5 \
  --env NCCL_IB_DISABLE=1 \
  --env NCCL_P2P_DISABLE=1 \
  --env NCCL_NET=Socket \
  --env NCCL_PORT=29501 \
  --env NCCL_DEBUG=INFO \
  --env NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH \
  --env NCCL_ASYNC_ERROR_HANDLING=1 \
  --env NCCL_BLOCKING_WAIT=1 \
  --env NCCL_TIMEOUT=1800 \
  --env NCCL_SOCKET_NTHREADS=4 \
  --env NCCL_NSOCKS_PERTHREAD=4 \
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
  --env NCCL_FINAL_CHECKPOINT_DIR="$runtime_root/final-checkpoints" \
  --volume "$base_root:$base_root" \
  "$image" \
  .venv/bin/torchrun \
    --nnodes="$world_size" \
    --nproc-per-node=1 \
    --node-rank="$rank" \
    --master-addr="$master_addr" \
    --master-port=29500 \
    "$base_root/capture_final.py" \
    --experiment "$base_root/frozen/experiment.yaml" \
    --profile official \
    --world-size "$world_size" \
    --seed "$seed" \
    --backend nccl \
    --dataset-cache "$runtime_root/cifar-cache" \
    --output-dir "$runtime_root/reference-output"
status=$?
printf '%s\n' "$status" >"$runtime_root/exit-code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$runtime_root/ended-at.txt"
exit "$status"
