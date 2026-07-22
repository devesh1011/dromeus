# syntax=docker/dockerfile:1

FROM golang:1.25-bookworm AS axl-builder

ARG AXL_GIT_COMMIT=628e28ace077f26dfe8d0259009b357216a9d8d4
RUN git clone https://github.com/gensyn-ai/axl.git /tmp/axl \
    && git -C /tmp/axl checkout "$AXL_GIT_COMMIT" \
    && cd /tmp/axl \
    && CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags='-s -w' \
       -o /opt/axl-node ./cmd/node

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY demo ./demo
RUN uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python \
       "msgpack>=1.1,<2" "numpy>=2,<3" "pydantic>=2.11,<3" \
       "pyyaml>=6.0,<7" "safetensors>=0.5,<1"
COPY --from=axl-builder /opt/axl-node /opt/axl/node

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app" \
    PYTHONUNBUFFERED=1 \
    DROMEUS_AXL_BINARY=/opt/axl/node

CMD ["python", "-m", "demo.formation.worker"]
