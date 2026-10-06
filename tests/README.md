# Test layout

Run the complete gate from the repository root:

```bash
./scripts/verify
```

Run a focused group with `uv run pytest tests/unit/transport -q`, or run the
benchmark checks with `uv run pytest tests/benchmarks -q`.

| Directory | Responsibility |
| --- | --- |
| `unit/algorithms` | NoLoCo/D-PSGD math, update bundles, codec contracts and compression |
| `unit/manifests` | Shared contracts, canonical identity and NoLoCo default selection |
| `unit/membership` | Formation and readiness with deterministic in-memory transport |
| `unit/runtime` | Node lifecycle, custom applications and local workload composition |
| `unit/training` | Generic and optional classification adapter behavior |
| `unit/gossip` | Generic rounds, concrete AXL exchange behavior with fakes, and convergence checks |
| `unit/transport` | Receiver, outbound scheduling, chunk transfer and transport adapters |
| `unit/protocol` | Envelope validation and frozen wire contracts |
| `unit/persistence` | Durable run state, archives and checkpoint validation |
| `unit/telemetry` | Events, metrics, evidence and consensus sketches |
| `benchmarks` | Workload models/data, reference implementations, experiment preparation and reporting |
| `integration` | Opt-in tests using real local AXL processes |
| `support` | Shared fakes, small input builders and stable repository/fixture paths |
| `golden` | Retained manifest/protocol bytes and independent NoLoCo reference outputs |

In-memory multi-node tests belong with the module whose behavior they exercise.
Real AXL integration tests require the environment variables documented in their
skip reasons. Real CIFAR-cache and reference checks may also be environment-gated;
a skipped case is not evidence that the corresponding deployment was exercised.

Benchmark coverage remains necessary even though its workload code is excluded
from the runtime wheel. Tests and test support are also excluded from that wheel.

Put new cases with their owning module. Share substantial reusable fakes through
`support`; keep one-off helpers close to their tests. Use `support.paths` for
repository and golden-fixture paths so moving tests does not break references.
Generated `__pycache__` files are disposable and already ignored by Git.

## M3 private training

Install the optional profile with `uv sync --frozen --extra privacy`. Run the local
contract and conformance checks with:

```bash
uv run --frozen --no-sync pytest tests/unit/training/test_private_trainer.py tests/unit/runtime/test_private_application.py tests/unit/persistence/test_privacy_ledger.py tests/unit/manifests/test_privacy_policy.py -q
```

The secure tests additionally require the pinned W0 source-built `torchcsprng`.
The privacy CI job covers public synthetic data and explicitly skips secure RNG
when the source-built dependency is unavailable; that skip does not certify secure
mode. Ordinary CI remains independent of Opacus.

Run the real four-process local AXL gate for SGD/Nesterov, Adam and AdamW with:

```bash
DROMEUS_PRIVATE_AXL_BINARY=/absolute/path/to/axl-node uv run --frozen --no-sync pytest tests/integration/test_private_training_axl.py -q
```

Set `DROMEUS_W1_EVIDENCE_ROOT` to retain compact summaries. The smoke factory uses
public synthetic data only and creates new synthetic ledgers. It is not a recipe
for recreating a missing private-data ledger. See [private training usage](../examples/private-training.md).

## M3 divergence monitoring

The W2 detector and runtime seams are covered by:

```bash
uv run --no-sync pytest tests/unit/telemetry tests/unit/runtime/test_divergence_runtime.py -q
```

The suite checks plateau/growth/severity, warm-up, patience, recovery, scale collapse,
reordering/duplicates/gaps, bounded state, strict v2 evidence and four runtime nodes
with normal, missing, blocked and failed telemetry. The blocked sink holds its lock
during warning I/O to exercise contention with other event producers.

`benchmarks/m3/replay_divergence.py` replays accepted M2 normalized observations
without rewriting them. Its explicit candidate thresholds are uncalibrated and the
report records false warnings and unavailable absolute-scale diagnostics. W3 owns
threshold calibration; see `examples/divergence-monitoring.md` for policy semantics.
