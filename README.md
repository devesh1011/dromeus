# Dromeus

Dromeus is a Python library for decentralized federated learning over
[AXL](https://github.com/gensyn-ai/axl). Each participant trains on local data and
averages model weights with randomly selected peers; raw training data stays on the
machine that owns it. Training nodes are symmetric and there is no central
coordinator.

## Development

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
./scripts/bootstrap
./scripts/verify
```

These are the canonical setup and verification commands used by CI. Verification
runs lockfile, architecture, lint, type, and test checks.

AXL is a separately managed runtime dependency. Importing `dromeus` does not start
AXL or perform any other runtime work.

## Run the nodes in Docker

Docker Compose runs four worker containers plus the dashboard. This is the easiest
path for another developer to reproduce the demo without installing Go locally.

```bash
docker compose -f demo/compose.yaml up --build -d
docker compose -f demo/compose.yaml ps
open http://127.0.0.1:8765
```

On Linux, replace `open` with `xdg-open`; or navigate to the URL manually. Start the
formation in the browser, or run:

```bash
curl -X POST http://127.0.0.1:8765/api/start
curl http://127.0.0.1:8765/api/state
docker compose -f demo/compose.yaml logs -f node-0 node-1 node-2 node-3
```

Cleanly stop the demo and remove its identities/artifacts with:

```bash
docker compose -f demo/compose.yaml down -v
```

More container/log commands are in [`demo/README.md`](demo/README.md).