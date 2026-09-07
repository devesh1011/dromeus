#!/usr/bin/env bash
set -uo pipefail

runtime_root="$1"
rank="$2"
base_root="/home/ubuntu/dromeus-official"
image="150911080841.dkr.ecr.us-east-1.amazonaws.com/dromeus-m2-benchmark@sha256:1b40d3774dc864f1f4450720f91243f2e379de8ec7d12805d120ec1ef1419a3c"

rm -f "${runtime_root}/exit-rank-${rank}" "${runtime_root}/ended-rank-${rank}.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"${runtime_root}/started-rank-${rank}.txt"
sudo docker run --rm \
  --gpus all \
  --network host \
  --shm-size 1g \
  --env PYTHONUNBUFFERED=1 \
  --volume "${base_root}:${base_root}" \
  "$image" \
  .venv/bin/python -m benchmarks.noloco.dromeus_official_runner \
  --node-config "${runtime_root}/node-${rank}.yaml" \
  --experiment "${base_root}/frozen-topk45-official/experiment.yaml" \
  --world-size 8 \
  --seed 29 \
  --output-dir "${runtime_root}/analysis/rank-${rank}"
status=$?
printf '%s\n' "$status" >"${runtime_root}/exit-rank-${rank}"
date -u +%Y-%m-%dT%H:%M:%SZ >"${runtime_root}/ended-rank-${rank}.txt"
exit "$status"
