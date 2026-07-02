#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/axl.env"

arch="${1:-amd64}"

case "$arch" in
  amd64)
    expected="$AXL_BINARY_SHA256_LINUX_AMD64"
    ;;
  arm64)
    expected="$AXL_BINARY_SHA256_LINUX_ARM64"
    ;;
  *)
    echo "unsupported arch: $arch" >&2
    exit 1
    ;;
esac

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT INT TERM

git clone "$AXL_GIT_URL" "$tmpdir/axl" >/dev/null
cd "$tmpdir/axl"
git checkout "$AXL_GIT_COMMIT" >/dev/null
CGO_ENABLED=0 GOOS=linux GOARCH="$arch" go build -trimpath -ldflags='-s -w' -o node ./cmd/node
actual=$(shasum -a 256 node | awk '{print $1}')

if [ "$actual" != "$expected" ]; then
  echo "AXL hash mismatch for linux/$arch: expected $expected got $actual" >&2
  exit 1
fi

echo "verified linux/$arch $AXL_GIT_COMMIT $actual"
