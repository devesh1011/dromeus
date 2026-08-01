# Dromeus

Dromeus is a Python 3.12 library for decentralized federated learning over
[AXL](https://github.com/gensyn-ai/axl).

The M1 release runs decentralized parallel SGD (D-PSGD) on four fixed nodes. Each
node trains on its own CIFAR-10 partition and exchanges model updates with one random
peer.

## Project status

M1 is complete and released as [`v0.1.0`](https://github.com/devesh1011/dromeus/releases/tag/v0.1.0). The release contains the runtime, benchmark report, charts, manifests, FedAvg controls, 12 final
checkpoints and public per-node training logs

### Results

Three frozen AWS D-PSGD seeds completed 400 training rounds and two final consensus
rounds. All 12 workers exited successfully after committed round `401`.

| Seed | D-PSGD | FedAvg |
| ---: | ---: | ---: |
| 17 | 91.69% | 91.71% |
| 29 | 91.44% | 91.55% |
| 41 | 91.60% | 91.23% |
| Mean | 91.5767% | 91.4967% |

D-PSGD finished 0.0800 percentage points above the FedAvg mean, so the acceptance
comparison passed. The D-PSGD standard deviation was 0.1034 percentage points. The
FedAvg standard deviation was 0.1996 percentage points.

Each node completed 16,000 local optimizer steps. Final consensus distances were
`0.0` for seed 17, `0.002431` for seed 29, and `0.0` for seed 41.

The all-pairs AXL baseline completed 300 transfers across 12 directed worker paths.
Every artifact passed checksum validation. Checkpoint transfers had zero retries.

The final artifacts are in
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
2. Four participants join and receive stable node indices.
3. The initiator seals a canonical manifest with the membership, model, dataset,
   optimizer, schedule, transport limits, and hashes.
4. Every node checks the manifest, environment, dataset, and initial checkpoint.
5. Once all nodes are ready, the initiator broadcasts `START`.

For every training round, all nodes derive the same seeded random matching. Each
pair then:

1. runs its local optimizer steps;
2. exchanges an opaque update bundle over AXL;
3. checks the peer, round, schema, size, checksum, and tensor values;
4. applies the D-PSGD mix:

   ```text
   next_model = 0.5 * local_update + 0.5 * peer_update
   ```

5. saves the next algorithm state and completes the pair commit handshake.

Pairs do not wait for the whole group. A pair that finishes can start its next round
while another pair is still working. A lost, invalid, or timed-out peer causes a
bounded failure. Dromeus does not silently rematch peers, shrink the group, or use
stale updates.

## Implemented in M1

- Immutable typed models for run configuration, membership, environment, dataset,
  training policy, and transport limits.
- Canonical JSON encoding and SHA-256 manifest identity.
- Fixed-membership formation over AXL, including invitations, joins, sealing,
  initial artifact distribution, environment checks, readiness, and start.
- Pinned Hugging Face CIFAR-10 loading, deterministic four-way IID partitions,
  preprocessing and augmentation, ResNet-32, CPU PyTorch training, SGD with
  momentum, and `safetensors` checkpoints.
- Pairwise D-PSGD with deterministic peer scheduling, opaque update bundles, FP32
  averaging, tensor validation, and serializable trainer and algorithm state.
- Reliable application-level AXL delivery with one receiver, bounded queues,
  control priority, per-peer lanes, retries, acknowledgments, deduplication,
  ordered transfers, cancellation safety, and checksums.
- Crash-safe persistence with one prepared candidate and one committed checkpoint.
  Archive version 2 is writable. Older v0 and v1 archives remain readable and
  read-only.
- Append-only JSONL diagnostics and typed evidence for node readiness, round metrics,
  consensus distance, transfer timing, and failures.
- Non-blocking 4,096-value CountSketch consensus telemetry. Reports use typed JSONL
  evidence and CountSketch trends, so they do not need every round's model snapshot.
- Matched FedAvg controls, frozen benchmark-plan validation, report generation,
  charts, and an all-pairs AXL transport baseline.
- A four-worker Docker Compose demo with one AXL node per worker and a dashboard for
  formation, training, state, and live metrics.

## Source modules

Each module owns one part of the system. Training does not depend on transport, and
transport does not know membership or model mathematics.

| Module | Purpose |
| --- | --- |
| `protocol` | Versioned wire models and bounded MessagePack encoding and decoding. |
| `manifests` | Run models, canonical JSON, validation, and hashing. |
| `membership` | Four-node formation, invitations, joins, sealed manifests, and readiness. |
| `transport` | AXL adapter, transport interface, receiver, outbound scheduler, retries, and artifact transfer. |
| `training` | Local data, deterministic partitions, ResNet-32, optimizer state, evaluation, and checkpoints. |
| `algorithms` | Algorithm and codec interfaces plus the M1 D-PSGD implementation. |
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
  training/         CIFAR-10 data, ResNet-32, trainer, checkpoints
  algorithms/       algorithm and codec interfaces; D-PSGD
  gossip/           peer scheduler and gossip engine
  persistence/      run store and archive reader
  telemetry/        events, evidence, metrics, and consensus
  runtime.py        lifecycle composition
  node.py           non-interactive AXL-backed node

benchmarks/cifar10/  D-PSGD/FedAvg runners, reports, plots, and AXL baselines
benchmarks/results/  archived local and AWS artifacts
demo/                Docker workers, AXL setup, CLI, and dashboard
tests/               unit, protocol, integration, and benchmark tests
scripts/             bootstrap, architecture checks, and verification gate
```

## Setup and verification

Use `uv`. Do not use `pip` or `conda`.

```bash
./scripts/bootstrap
./scripts/verify
```

The verification gate checks the lockfile, dependency direction, production
isolation, the single-receiver and transfer-opacity rules, cycle freedom, Ruff,
strict Pyright, and pytest. The final gate passed with `201 passed, 2 skipped`.
Those two skipped tests are opt-in real-AXL integration tests.

To run local real-AXL integration tests, provide a local AXL setup and opt in:

```bash
DROMEUS_RUN_AXL_TESTS=1 uv run pytest tests/integration/test_local_axl_formation.py -q
```

## Reproduce a local training run with Docker

The demo starts four independent Dromeus workers, each with its own AXL identity,
plus a dashboard.

This is the recommended way to reproduce the training flow. It runs four workers
and four AXL nodes on one machine. It checks formation, real AXL messaging, local
training, pairwise mixing, persistence, and telemetry without requiring AWS.

```bash
docker compose -f demo/compose.yaml up --build -d
docker compose -f demo/compose.yaml ps
open http://127.0.0.1:8765
```

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

## Benchmark tooling

The benchmark controller lives outside `src/dromeus`. It supports:

- frozen plans and configuration validation;
- local dataset proofs and worker configuration;
- real-AXL D-PSGD execution;
- matched centralized FedAvg runs;
- strict per-seed and aggregate evidence and report validation;
- accuracy, loss, timing, goodput, schedule, retry, failure, and consensus plots;
- distributed all-pairs AXL RTT and artifact-transfer baselines.

```bash
uv run python -m benchmarks.cifar10.runner --help
```

The quality recipe uses four nodes, 400 training rounds, 40 local steps per round,
batch size 128, CPU PyTorch, deterministic seeds, and two final non-training
consensus rounds. The sealed manifest and archived evidence contain the exact
settings and hashes for each run.

The demo uses a lighter configuration than the AWS benchmark: one local step per
round, learning rate `0.01`, batch size `16`, and no final consensus rounds. You can
run up to 100 rounds with the demo. For example:

```bash
uv run python -m demo.formation.cli form --rounds 100 --wait
uv run python -m demo.formation.cli train --follow
```

This reproduces the distributed training behavior. It is not intended to reproduce
the final AWS accuracy number.

## M1 release artifacts

- [GitHub Release v0.1.0](https://github.com/devesh1011/dromeus/releases/tag/v0.1.0)
  contains the public benchmark ZIP and the standalone Gensyn submission DOCX.
- The benchmark package contains the final report, charts, manifests, FedAvg
  controls, and 12 final `safetensors` checkpoints.
- Public per-node logs contain the 12 D-PSGD `dromeus.jsonl` files.
