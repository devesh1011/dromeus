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
| `demo` | Demo-specific behavior |
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
