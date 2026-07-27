# syntax=docker/dockerfile:1

FROM golang:1.25-bookworm AS axl-builder

ARG AXL_GIT_COMMIT=628e28ace077f26dfe8d0259009b357216a9d8d4
ARG AXL_BINARY_SHA256=af914d445ff16f00a70444342e23a226a6e2f3ca8bcabf8d9390d59cae5681e7

RUN git clone https://github.com/gensyn-ai/axl.git /tmp/axl \
    && git -C /tmp/axl checkout "$AXL_GIT_COMMIT" \
    && cd /tmp/axl \
    && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
       -trimpath -ldflags='-s -w' -o /opt/axl/node ./cmd/node \
    && echo "$AXL_BINARY_SHA256  /opt/axl/node" | sha256sum --check

FROM python:3.12-slim-bookworm

ARG DROMEUS_COMMIT
ARG AXL_GIT_COMMIT=628e28ace077f26dfe8d0259009b357216a9d8d4
ARG AXL_BINARY_SHA256=af914d445ff16f00a70444342e23a226a6e2f3ca8bcabf8d9390d59cae5681e7

COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY benchmarks ./benchmarks
RUN uv sync --frozen --no-dev \
    && printf \
       '{"binary_sha256":"%s","source_commit":"%s"}\n' \
       "$AXL_BINARY_SHA256" "$AXL_GIT_COMMIT" \
       > /opt/axl-build.json

COPY --from=axl-builder /opt/axl/node /opt/axl/node

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app" \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.revision="$DROMEUS_COMMIT"

CMD ["python", "-m", "dromeus.node", "--help"]
