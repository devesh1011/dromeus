#!/bin/sh
set -eu

# Docker Desktop's Rosetta binfmt handler adds this private switch when an
# amd64 executable is started on an arm64 host. AXL does not define it.
if [ "${1:-}" = "-no-opt" ]; then
    shift
fi
exec /opt/axl/node "$@"
