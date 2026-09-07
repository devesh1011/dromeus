# Dromeus

Dromeus is a Python 3.12 library for **federated learning with NoLoCo over
[AXL](https://github.com/gensyn-ai/axl)**. NoLoCo is the default decentralized
algorithm; developers supply their own models, local data, and local optimizer
steps. Dromeus handles NoLoCo peer updates and communication.

The M1 release runs decentralized parallel SGD (D-PSGD) on four fixed nodes. Each
node trains on its own CIFAR-10 partition and exchanges model updates with one random
peer.

## M1 release status

The M1 implementation is complete and published as
[`v0.1.0`](https://github.com/devesh1011/dromeus/releases/tag/v0.1.0). External
milestone acceptance is pending. The repository and release include the runtime,
benchmark report, charts, manifests, FedAvg controls, final checkpoints, and the
Gensyn submission brief. Public per-node logs are published separately.

### M1 benchmark summary

The official benchmark used four AWS training nodes and three frozen seeds, `17`,
`29`, and `41`. Each run completed 400 D-PSGD training rounds and two final
consensus rounds. All 12 D-PSGD workers completed successfully.

D-PSGD reached a mean accuracy of `91.5767%`, compared with `91.4967%` for
matched FedAvg. The `0.0800` percentage-point benchmark comparison passed.

The all-pairs AXL baseline completed 300 transfers across 12 directed paths. Every
artifact passed checksum validation, and checkpoint transfers had zero retries.

Detailed per-seed metrics, charts, logs, and benchmark evidence are in the
[GitHub release](https://github.com/devesh1011/dromeus/releases/tag/v0.1.0) and the
archived artifacts at
[`benchmarks/results/dromeus-m1-20260731/`](benchmarks/results/dromeus-m1-20260731/).

## System model

AXL runs as a separate process. Each Dromeus process talks to its local AXL HTTP
bridge. AXL handles encrypted routing through the mesh.

```text
Node A                                      Node B
┌──────────────────────┐                    ┌──────────────────────┐
│ Dromeus              │                    │ Dromeus              │
│ train → exchange     │                    │ train → exchange     │
│ validate → mix       │                    │ validate → mix       │
└──────────┬───────────┘                    └──────────┬───────────┘
           │ localhost HTTP                            │ localhost HTTP
     ┌─────▼─────┐                                ┌────▼──────┐
     │ AXL node  │◄──── encrypted AXL mesh ─────► │ AXL node  │
     └───────────┘                                └───────────┘
```

## Run lifecycle

1. The initiator publishes a draft run specification and invitation.
2. Participants join the initiator, forming a fixed even group of 4–16 nodes
   with stable node indices.
3. The initiator seals a canonical manifest with the membership, model, dataset,
   optimizer, schedule, transport limits, and hashes.
4. Every node checks the manifest, environment, dataset, and initial checkpoint.
5. Once all nodes are ready, the initiator broadcasts `START`.

For every training round, all nodes derive the same seeded random matching. Each
pair then:

1. starts from its NoLoCo slow weights and runs the application's local steps;
2. forms the outer gradient and exchanges it with the slow weights over AXL,
   using the declared codecs and error feedback for lossy outer gradients;
3. checks the peer, round, schema, size, checksum, and tensor values;
4. applies NoLoCo's outer momentum and slow-weight correction;
5. saves the next algorithm state and completes the pair commit handshake.

Drafts default to `algorithm_id: noloco`. The sealed manifest records that
selection explicitly. D-PSGD remains available through an explicit
`algorithm_id: dpsgd` selection for compatibility and controls. Local optimizers
such as Adam or RMSprop operate inside NoLoCo's inner steps.

Pairs do not wait for the whole group. A pair that finishes can start its next round
while another pair is still working. A lost, invalid, or timed-out peer causes a
bounded failure. Dromeus does not silently rematch peers, shrink the group, or use
stale updates.

## Source modules

Each module owns one part of the system. Training does not depend on transport, and
transport does not know membership or model mathematics.

| Module | Purpose |
| --- | --- |
| `protocol` | Versioned wire models and bounded MessagePack encoding and decoding. |
| `manifests` | Run models, canonical JSON, validation, and hashing. |
| `membership` | Fixed 4–16-node formation, invitations, joins, sealed manifests, and readiness. |
| `transport` | AXL adapter, transport interface, receiver, outbound scheduler, retries, and artifact transfer. |
| `training` | Application-owned training interface, tensor state, named evaluation metrics, and checkpoints. |
| `application` / `adapters` | Custom workload composition and the optional classification recipe. |
| `algorithms` | NoLoCo outer optimization, explicit D-PSGD compatibility, and update codecs. |
| `gossip` | Peer matching, event-driven rounds, bundle exchange, validation, and pair commit. |
| `runtime` / `node` | Lifecycle composition and the non-interactive node entry point. |
| `persistence` | Atomic `RunStore`, validated `RunArchive`, checkpoint references, and terminal state. |
| `telemetry` | JSONL diagnostics, typed evidence, metrics, CountSketch, and consensus reporting. |

## Repository layout

```text
src/dromeus/
  protocol/         wire models, protocol version, bounded MessagePack codec
  manifests/        domain models and canonical encoding
  membership/       fixed-group formation
  transport/        AXL adapter, receiver, scheduler, and transfers
  training/         custom training interface, tensor state, checkpoints
  adapters/         optional classification training and data validation
  application.py    developer-owned workload composition
  algorithms/       NoLoCo, D-PSGD compatibility, and codecs
  gossip/           peer scheduler and gossip engine
  persistence/      run store and archive reader
  telemetry/        events, evidence, metrics, and consensus
  runtime.py        lifecycle composition
  node.py           non-interactive AXL-backed node

benchmarks/workloads/cifar10/  CIFAR loaders, ResNet models, benchmark registry
benchmarks/cifar10/  D-PSGD/FedAvg runners, reports, plots, and AXL baselines
examples/            custom regression model, local factory, and configuration
benchmarks/results/  archived local and AWS artifacts
demo/                Docker workers, AXL setup, CLI, and dashboard
tests/               module-grouped unit tests, benchmark checks, real AXL integration
scripts/             bootstrap, architecture checks, and verification gate
```

## Train your own model

Supply your own local Python factory to `dromeus.node.run_node`, or launch it with
`python -m dromeus.node --config node.yaml --factory your_package:prepare_training`.
The factory owns the model, optimizer, loss, data loading, and optional evaluation.
Each participant uses its own local data under the group's shared task and tensor
contract. See the [custom regression example and migration guide](examples/README.md).

CIFAR-10 and ResNet recipes live in `benchmarks/workloads/cifar10` and are excluded
from the installed wheel. The benchmark runners explicitly select those workloads.
`datasets` and Pillow are benchmark dependencies, available via the `benchmark`
extra and included in the repository development environment.

## Setup and verification

Use `uv`. Do not use `pip` or `conda`.

```bash
./scripts/bootstrap
./scripts/verify
```

See [the test layout](tests/README.md) for focused commands and test ownership.

The verification gate checks the lockfile, dependency direction, production
isolation, the single-receiver and transfer-opacity rules, cycle freedom, Ruff,
strict Pyright, and pytest. Environment-gated tests report their prerequisites
in the skip reason; canonical milestone status is maintained in Obsidian.

To run local real-AXL integration tests, provide a local AXL setup and opt in:

```bash
DROMEUS_RUN_AXL_TESTS=1 uv run pytest tests/integration/test_local_axl_formation.py -q
```

## Reproduce the M1 demo with Docker

The demo starts four independent Dromeus workers, each with its own AXL identity,
plus a dashboard.

This historical CIFAR/D-PSGD demo runs four workers and four AXL nodes on one
machine. For custom NoLoCo training, use the [application guide](examples/README.md).
The demo checks formation, real AXL messaging, local
training, pairwise mixing, persistence, and telemetry without requiring AWS.

```bash
docker compose -f demo/compose.yaml up --build -d
docker compose -f demo/compose.yaml ps
```

Open `http://127.0.0.1:8765` in a browser.

The Compose file targets `linux/amd64`. Docker Desktop runs it through emulation
on Apple Silicon, so the first build and training run may be slower.

Form the group and start a short run from the demo CLI:

```bash
uv run python -m demo.formation.cli form --rounds 5 --wait
uv run python -m demo.formation.cli train --follow
```

Or use the dashboard API:

```bash
curl -X POST http://127.0.0.1:8765/api/start \
  -H 'Content-Type: application/json' \
  -d '{"round_count": 5}'
curl http://127.0.0.1:8765/api/state
docker compose -f demo/compose.yaml logs -f node-0 node-1 node-2 node-3
```

Stop the demo and remove its identities and artifacts:

```bash
docker compose -f demo/compose.yaml down -v
```

See [`demo/README.md`](demo/README.md) for more commands.

## M1 release artifacts

- [GitHub Release v0.1.0](https://github.com/devesh1011/dromeus/releases/tag/v0.1.0)
  contains the public benchmark ZIP and the standalone Gensyn submission DOCX.
- The benchmark package contains the final report, charts, manifests, FedAvg
  controls, and 12 final `safetensors` checkpoints.
- Public per-node logs contain the 12 D-PSGD `dromeus.jsonl` files. The repository
  includes the downloaded seed-17 logs under `official/logs/seed-17/`.
