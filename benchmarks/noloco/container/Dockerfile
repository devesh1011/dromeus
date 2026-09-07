FROM ghcr.io/astral-sh/uv@sha256:78a2f9fd8f605301464fc61c42be3acb5a65cd92b0c91f481c3dcdd411dff810

WORKDIR /workspace/dromeus

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra benchmark --no-dev --no-install-project --no-install-package torch

COPY src/ src/
COPY benchmarks/ benchmarks/
RUN uv sync --frozen --extra benchmark --no-dev --no-install-package torch \
    && uv pip install --python .venv/bin/python \
        torch==2.12.1 \
        --index-url https://download.pytorch.org/whl/cu130

CMD [".venv/bin/python", "-m", "benchmarks.noloco.container_preflight"]
