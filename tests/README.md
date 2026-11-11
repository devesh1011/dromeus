# Test layout

Run the complete gate from the repository root:

```bash
./scripts/verify
```

It checks the lockfile, architecture boundaries (`scripts/check_structure.py`),
lint, strict types and the test suite. Run one area with, for example,
`uv run pytest tests/unit/transport -q`.

| Directory | Responsibility |
| --- | --- |
| `unit/algorithms` | NoLoCo outer update, update bundles, codec contracts and compression |
| `unit/manifests` | Manifest v7 contract, canonical identity, task/privacy/divergence policies |
| `unit/membership` | Formation and readiness with deterministic in-memory transport |
| `unit/runtime` | Node lifecycle, application and task preparation, private preflight |
| `unit/training` | Task trainers (ordinary and private), optimizers, private model checks |
| `unit/gossip` | Rounds, pair exchange over fakes, scheduling and convergence |
| `unit/transport` | Receiver, outbound scheduling and chunked transfer |
| `unit/protocol` | Envelope validation and the frozen wire contract |
| `unit/persistence` | Run store, archives, durable writes and the privacy ledger |
| `unit/telemetry` | Events, metrics, evidence records, consensus sketches, divergence detection |
| `integration` | Opt-in tests that start real local AXL processes |
| `support` | Shared fakes, fixtures, the local AXL harness and stable paths |
| `golden` | Frozen manifest and protocol bytes, and the independent NoLoCo reference step |

Put new cases with the module whose behaviour they exercise. Share substantial
fakes through `support`; keep one-off helpers next to their tests. Use
`support.paths` for repository and golden-fixture paths. The example application
in `examples/custom_training.py` doubles as the custom-training fixture, and
`unit/runtime/test_example_application.py` keeps the published example draft valid.

## Private training

Private tests need the optional profile:

```bash
uv sync --frozen --extra privacy
uv run --frozen --no-sync pytest tests/unit/training tests/unit/runtime/test_private_application.py tests/unit/runtime/test_factory_private_application.py tests/unit/persistence/test_privacy_ledger.py tests/unit/manifests/test_privacy_policy.py -q
```

The CI `privacy` job runs these checks on public synthetic data. Secure-RNG cases
also need a source-built `torchcsprng`; they are skipped when it is missing, and a
skip does not certify secure mode.

## Real AXL integration

Integration tests skip unless their environment is set. A skip is not evidence
that the deployment path was exercised.

```bash
# Harness, custom-application and private-task runs over four local AXL nodes
DROMEUS_AXL_BINARY=/absolute/path/to/axl-node \
uv run --frozen --no-sync pytest tests/integration -q

# Formation and 45 MiB multi-chunk transfer (builds the pinned AXL revision)
DROMEUS_RUN_AXL_TESTS=1 uv run --frozen --no-sync pytest tests/integration/test_local_axl_formation.py -q
```

Set `DROMEUS_PRIVATE_EVIDENCE_ROOT` to keep compact summaries from the private run.
It uses public synthetic data and fresh synthetic privacy ledgers.

## Past milestones

Benchmarks and evidence for earlier milestones are verified at their release tags
(`v0.1.0`, `v0.2.0`, `v0.3.0`), not on `main`. Retained results under
`benchmarks/results/` are frozen and excluded from type checking.
