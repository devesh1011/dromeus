# Dromeus

Dromeus is a Python library for decentralized federated learning over
[AXL](https://github.com/gensyn-ai/axl). It implements the first Dromeus milestone:
four fixed, symmetric training nodes running pairwise decentralized parallel SGD
(D-PSGD).

There is no central coordinator or global model owner. Each node trains on its own
local CIFAR-10 partition and exchanges model updates only with a scheduled peer.
Raw training data never leaves the node that owns it.

## Current status

The M1 runtime, training path, reliability layer, persistence, telemetry, benchmark
tools, and reproducible Docker demo are implemented.

The 400-round AWS run completed on all four nodes. Every node passed 91.6%
accuracy by round 324. Each node wrote a `committed-round-000401.safetensors`
checkpoint with 16,000 completed optimizer steps (400 rounds × 40 local steps).
These files confirm that training finished. The aggregate report is separate and
still needs to be generated.

## How a run works

AXL is a separately managed process. Dromeus talks only to the local AXL HTTP
bridge; AXL handles encrypted peer-to-peer routing through its mesh.

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

At startup:

1. The initiator publishes a draft run specification and invitation.
2. Exactly four participants join and receive stable node indices.
3. The initiator seals a canonical manifest containing membership, model, dataset,
   optimizer, schedule, transport limits, and hashes.
4. Every node validates the manifest, environment, dataset, and initial checkpoint.
5. After all nodes are ready, training starts.

For each training round, every node independently derives the same seeded random
perfect matching. Each pair:

1. runs local optimizer steps;
2. exchanges an opaque update bundle over reliable AXL transfer;
3. validates sender, round, schema, size, checksum, and finite tensor values;
4. applies the D-PSGD mix:

   ```text
   next_model = 0.5 * local_update + 0.5 * peer_update
   ```

5. persists the next state and completes the pair commit handshake.

There is no group-wide barrier after training begins. A committed pair can move to
the next round while another pair is still working. A lost or invalid peer causes a
bounded, explicit run failure; the group is not silently rematched or shrunk.

## What is implemented

- Typed, immutable protocol models with canonical JSON encoding and SHA-256 manifest
  identity.
- Fixed-membership formation over AXL: invitations, join requests, sealing, readiness,
  start, sender allowlists, and environment validation.
- Local CIFAR-10 training using a pinned Hugging Face dataset revision, deterministic
  four-way IID partitions, a versioned ResNet-32 recipe, CPU PyTorch training, data
  augmentation, checkpoint hashing, and `safetensors` artifacts.
- Pairwise D-PSGD with configurable local steps, deterministic peer scheduling, FP32
  50/50 mixing, validated update bundles, and serializable algorithm state.
- Reliable application-level delivery over AXL: one receiver, bounded queues,
  control-message priority, retries, acknowledgments, deduplication, ordered
  transfers, cancellation safety, and integrity checks.
- Atomic run persistence after each committed round, including checkpoints, manifest,
  state, schedule, metrics, transfer diagnostics, and terminal result.
- Non-blocking per-node JSONL events and live CountSketch consensus telemetry, plus
  exact offline consensus calculations from archived checkpoints.
- A four-worker Docker Compose demo with an AXL node per worker and a small dashboard
  for formation, training, state, and live metrics.

## `src/dromeus` modules

The source is split by ownership. Training mathematics does not depend on transport,
and transport does not know membership or model details.

| Module | Purpose |
| --- | --- |
| `manifests` | Defines run/configuration models; validates, canonicalizes, and hashes protocol data. |
| `membership` | Forms the fixed group, assigns node indices, distributes the initial artifact, and runs the ready/start barrier. |
| `transport` | Provides the byte-delivery seam and AXL adapter: envelopes, receiver routing, outbound scheduling, retries, and artifact transfer. |
| `training` | Owns local data, model construction, preprocessing, optimizer execution, evaluation, and checkpoints. Raw data stays behind this interface. |
| `algorithms` | Defines the algorithm/codec seam and implements M1 D-PSGD. Update bundles are encoded and validated here. |
| `gossip` | Schedules random peer matchings and drives event-driven rounds, update exchange, validation, and pair commit. |
| `runtime` / `node` | Composes the modules and owns the external lifecycle: form, ready, run, complete, or fail. |
| `persistence` | Atomically stores manifests, algorithm state, checkpoints, metrics, schedules, and failures. |
| `telemetry` | Emits observational events and metrics, including CountSketch consensus estimates. It never controls training. |

`InMemoryTransport` exists only for deterministic unit/protocol tests. Production and
milestone integration runs use `AXLTransport`.

## Repository layout

```text
src/dromeus/
  algorithms/       algorithm and codec interfaces; D-PSGD
  gossip/           peer scheduler and event-driven gossip engine
  manifests/        typed models and canonical encoding
  membership/       invitation, join, sealing, readiness
  persistence/      atomic run store
  telemetry/        events, metrics, consensus sketches, reports
  training/         CIFAR-10 data, model, trainer, checkpoints
  transport/        AXL adapter, envelopes, receiver, sender, transfers
  runtime.py        lifecycle composition
  node.py           non-interactive AXL-backed node entry point

benchmarks/cifar10/  FedAvg, benchmark runner, frozen plans, reports, plots
benchmarks/results/  archived local/AWS run artifacts and reports
demo/                Docker Compose workers, AXL setup, dashboard
tests/               unit, protocol, integration, and benchmark tests
scripts/             bootstrap, architecture check, and complete verification gate
```

## Setup and verification

Use `uv`; do not use `pip` or `conda`.

```bash
./scripts/bootstrap
./scripts/verify
```

`./scripts/verify` runs the lockfile check, architectural dependency checks, Ruff,
strict Pyright, and pytest. The latest recorded gate passed with 161 tests passed and
2 skipped. The skipped tests are opt-in real-AXL tests.

To run the local AXL integration tests when a local AXL setup is available:

```bash
DROMEUS_RUN_AXL_TESTS=1 uv run pytest tests/integration/test_local_axl_formation.py -q
```

Dromeus does not download, start, or supervise the AXL binary. Start a pinned AXL
node separately and provide its loopback bridge URL.

## Docker demo

The demo runs four independent workers, each with its own Dromeus process and AXL
identity, plus a dashboard. CIFAR-10 is downloaded into a shared Docker volume and
reused by subsequent runs.

```bash
docker compose -f demo/compose.yaml up --build -d
docker compose -f demo/compose.yaml ps
open http://127.0.0.1:8765
```

Choose the number of D-PSGD rounds in the dashboard, form the fixed group, then
start training. The same flow is available from the demo CLI:

```bash
uv run python -m demo.formation.cli form --rounds 5 --wait
uv run python -m demo.formation.cli train --follow
```

The dashboard API can also start and inspect a run:

```bash
curl -X POST http://127.0.0.1:8765/api/start \
  -H 'Content-Type: application/json' \
  -d '{"round_count": 5}'
curl http://127.0.0.1:8765/api/state
docker compose -f demo/compose.yaml logs -f node-0 node-1 node-2 node-3
```

Stop the demo and remove its demo-only identities/artifacts with:

```bash
docker compose -f demo/compose.yaml down -v
```

See [`demo/README.md`](demo/README.md) for additional dashboard and log commands.
