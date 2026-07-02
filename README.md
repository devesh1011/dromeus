# Dromeus

Dromeus is a Python library for decentralized federated learning over
[AXL](https://github.com/gensyn-ai/axl). Each participant trains on local data and
averages model weights with randomly selected peers; raw training data stays on the
machine that owns it. Training nodes are symmetric and there is no central
coordinator.

Milestone 1 builds a four-node D-PSGD benchmark on CIFAR-10. See
[`docs/architecture.md`](docs/architecture.md) for the design and
[`docs/m1-plan.md`](docs/m1-plan.md) for the implementation plan.

## Development

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen
uv run ruff check .
uv run pyright
uv run pytest
```

The internal node entry point used by containers and end-to-end tests is:

```bash
uv run python -m dromeus.node --config /path/to/node.yaml
```

AXL is a separately managed runtime dependency. Importing `dromeus` does not start
AXL or perform any other runtime work.

The pinned upstream AXL commit and expected Linux binary hashes live in
[`docker/axl.env`](docker/axl.env). Use [`docker/verify-axl.sh`](docker/verify-axl.sh)
to rebuild and verify that pinned binary during image builds.
