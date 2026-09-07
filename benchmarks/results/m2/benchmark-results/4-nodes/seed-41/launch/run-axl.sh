#!/usr/bin/env bash
set -euo pipefail

runtime_root="$1"
base_root="/home/ubuntu/dromeus-official"

rm -f "${runtime_root}/axl.log" "${runtime_root}/axl.pid" "${runtime_root}/axl-ended-at.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"${runtime_root}/axl-started-at.txt"
nohup "${base_root}/axl-node" -config "${runtime_root}/axl.json" \
  >"${runtime_root}/axl.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"${runtime_root}/axl.pid"
