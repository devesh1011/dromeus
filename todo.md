# Dromeus engineering follow-ups

Created 2026-09-06. The source baseline was committed locally on 2026-09-07 to
`codex/m2-benchmark-baseline`. **Implementation tasks remain pending.** Resume
those tasks when the user requests implementation.

This is the requested working backlog, not the canonical M2 milestone record.
Read the Obsidian [current progress](</Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/M2 Current Progress.md>),
[M2 plan](</Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/plans/M-2 implementation plan.md>),
and [architecture](</Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/Architecture.md>)
before implementing. Record project decisions and verified status there.

All nine compressed AXL runs at W4/W8/W16 and seeds 17/29/41 are accepted in the
current progress note. That note still lists paper-length NCCL references, the
four-node identity ablation, report, and release as open. These engineering
follow-ups do not mark those deliverables complete or change their acceptance gates.

## Recommended order

Preserve the benchmark baseline, fix receiver backpressure, implement local-data
support, and validate non-IID behavior. Codec cleanup and AXL adapter extraction
are independent maintenance work and can follow in separate changes.

## 0. Preserve reproducibility before code changes

- [x] Record the source snapshot, frozen experiment hashes, image digest, and
  evidence locations for the completed M2 runs. Keep new implementations distinct
  from the runtime that produced the accepted evidence.
- [x] Track the `benchmarks/noloco/` and
  `benchmarks/noloco_reference/` packages and required provenance tooling in the
  release source. Preserve unrelated working-tree edits.
- [ ] Keep existing manifest-v3 canonical bytes, codec formats, frozen partitions,
  optimizer settings, and official artifacts stable. Introduce an explicit
  version for changed data-contract semantics; decide the versioning approach
  before implementation.

The local baseline includes the benchmark harnesses, provenance tooling, and compact
accepted evidence. Raw logs, initial checkpoint binaries, and full per-node records
remain outside Git with their existing hash bindings. Public artifact distribution
and the remaining M2 deliverables are still open. Full verification of an export
of the staged source passed: 332 tests, 10 environment-gated skips, and clean
architecture, lint, and type checks.

**Done when:** the completed benchmark configuration can be reproduced from a
retained source snapshot, and new local-data experiments have separate identities.

## 1. Fix receiver backpressure — reliability priority

**Evidence:** [Receiver._route](/Users/devesh1011/code_with_devesh/projects/dromeus/src/dromeus/transport/receiver.py:223)
awaits admission to every channel queue. The telemetry queue has capacity 64.
An offline reproduction sent 65 valid telemetry messages from 15 sealed peers
across five rounds, followed by `RUN_FAILED`. The failure message was delivered
only after one telemetry item was consumed. This was a local reproduction, not an
observed failure of an accepted W16 benchmark.

- [ ] Make telemetry admission nonblocking. Select and document a bounded drop or
  coalescing policy; count discarded telemetry for diagnostics.
- [ ] Keep the single transport reader draining while the telemetry consumer is
  slow or stalled, so ACKs and control messages remain deliverable.
- [ ] Audit other full-channel paths and shutdown. Ensure stopping a receiver
  cannot wait indefinitely on a full queue; preserve required control/transfer
  semantics rather than silently dropping essential protocol messages.
- [ ] Add regressions for a full telemetry queue followed by `CHUNK_ACK`,
  `UPDATE_READY`, and `RUN_FAILED`, plus shutdown while a consumer is stalled.

**Done when:** the reproduction no longer blocks control traffic, queues remain
bounded, drop accounting is accurate, and transport/commit failure tests pass.

## 2. Support independently held local datasets — product priority

**Current restriction:** [DatasetContract](/Users/devesh1011/code_with_devesh/projects/dromeus/src/dromeus/manifests/models.py:76)
requires CIFAR-10 and a global IID partition map.
[CIFAR preparation](/Users/devesh1011/code_with_devesh/projects/dromeus/src/dromeus/training/cifar10.py:330)
loads the full common source before selecting a local partition.
[Formation](/Users/devesh1011/code_with_devesh/projects/dromeus/src/dromeus/membership/formation.py:54)
compares the complete declared dataset contract and derives membership size from
its partitions. The existing trainer accepts a PyTorch classification dataset,
and the runtime already has a training-factory seam.

### Shared task contract and membership

- [ ] Define a shared classification-task contract covering input shape/dtype,
  target representation, ordered label meanings, preprocessing identity, model
  definition/tensor schema, and compatible training policy.
- [ ] Keep local dataset paths, source identities, sample counts, and content
  fingerprints in a node-local binding. Different nodes may have different
  records, counts, label proportions, and absent classes.
- [ ] Declare fixed participant count independently of a global dataset or
  partition map. Retain sealed membership, deterministic node indices, and the
  existing formation sequence.
- [ ] Keep strict CIFAR/IID source and partition checks available for reproducing
  the frozen benchmark. Avoid making those checks prerequisites for every local
  dataset.
- [ ] Version the new contract explicitly and test parser rejection of invalid
  version/contract combinations alongside existing v3 golden hashes.

### Local data preparation and validation

- [ ] Add a training-owned interface for developer-supplied local PyTorch datasets
  or loader factories. Start with the classification workload the trainer supports;
  choose the first file-based adapter separately if one is needed.
- [ ] Wire local preparation into the node/runtime lifecycle so the local-data
  path does not download CIFAR or reconstruct another node's partition.
- [ ] Validate actual local data before `READY`: nonempty training data, valid
  input shape/dtype, finite values, legal target values, and matching declared
  label/preprocessing semantics. Missing local classes are allowed; conflicting
  meanings for the same label index reject.
- [ ] Bind local dataset identity and sampler/loader state to local checkpoint
  restoration. Define behavior when the local dataset changes; do not silently
  resume against different records or ordering.
- [ ] Require an explicit evaluation policy for the local-data path. Use local
  held-out data or mark evaluation unavailable; do not silently report training-set
  accuracy as held-out accuracy.

### Unequal dataset sizes and learning objective

- [ ] Document the current equal-node update: every node performs 50 local steps,
  and NoLoCo's pair gradient contribution does not use dataset sample counts.
- [ ] Decide whether the initial local-data feature retains equal-node weighting.
  If sample-weighted learning is required, specify and independently validate its
  algorithm rather than inserting an unvalidated weighted pair average.
- [ ] Define handling of tiny datasets, partial batches, and repeated passes during
  a local block. Reject empty datasets and unsupported policies before formation.

**Done when:** independently supplied, unequal-size classification datasets can
complete multi-round training through the real runtime/formation interfaces;
schema/label mismatches reject before readiness; raw data stays local; and the
existing CIFAR benchmark path remains reproducible.

## 3. Establish non-IID evidence

Loading independent datasets and obtaining useful learning under heterogeneous
distributions are separate acceptance questions. Current CIFAR benchmarks cover
disjoint IID partitions. The [NoLoCo paper's convergence analysis](https://arxiv.org/html/2506.10911v1)
assumes identically distributed outer gradients across replicas; it does not
establish arbitrary non-IID convergence for Dromeus's compressed adaptation.

- [ ] Add deterministic experimental data plans for an IID control, moderate and
  severe label skew, unequal sample counts, and feature/domain skew. Isolate each
  source of heterogeneity before combining them. Record seeds and actual partitions.
- [ ] Add a four-node integration case where each worker can access only its own
  data. Exercise both identity and compressed NoLoCo, unequal counts, and missing
  local classes through the public runtime interfaces.
- [ ] Compare identity and compressed NoLoCo, the independent reference using the
  same data plan, and local-only training. Declare the compute/sample budget used
  for comparisons.
- [ ] Report held-out loss/accuracy per node, mean and worst-node performance,
  per-class results where appropriate, residuals, weight divergence, communication,
  and round timing. Use a common public test set only when it is explicitly part
  of the experimental design.
- [ ] Freeze experiment-specific acceptance criteria before runs. Check learning
  utility as well as protocol completion and weight agreement; the IID M2 accuracy,
  residual, and divergence thresholds are not automatically non-IID guarantees.
- [ ] Start with local diagnostics and four-node real-AXL validation, then decide
  whether results justify W8/W16 runs. Label in-memory/Gloo checks as development
  evidence and keep these experiments separate from the accepted M2 matrix.
- [ ] If learning quality fails, use controlled ablations to test local-step
  interval, learning rate, outer settings, and compression. Record any new algorithm
  or policy decision before changing implementations or claiming support.

**Done when:** a retained report states which heterogeneous workloads pass the
predeclared criteria, where they fail, and whether compression changes the outcome.
Successful execution alone is not a non-IID learning claim.

## 4. Make codec capabilities explicit — maintenance

**Evidence:** [UpdateCodec](/Users/devesh1011/code_with_devesh/projects/dromeus/src/dromeus/algorithms/codec.py:46)
does not declare every capability consumed by
[NoLoCo](/Users/devesh1011/code_with_devesh/projects/dromeus/src/dromeus/algorithms/noloco.py:415).
NoLoCo probes `lossy`, `codec_version`, and `encoded_schema` with `getattr`, including
defaults that affect error feedback and metadata interpretation.

- [ ] Define these capabilities in the typed codec interface or an immutable codec
  description. Specify how identity codecs obtain their encoded schema.
- [ ] Migrate identity, dense-int8, top-k v1, and bitmap v2 implementations and
  algorithm callers; reject incomplete capabilities at construction/binding.
- [ ] Preserve existing codec IDs, versions, encoded bytes, deterministic selection,
  quantization, and error-feedback behavior.
- [ ] Verify all supported codecs through the common interface, including malformed
  metadata and schema rejection, identity parity, compressed traces, and restore.

**Done when:** capability discovery is explicit and type-checked, with all existing
wire-format and algorithm parity tests unchanged in meaning and passing.

## 5. Separate AXL exchange from round orchestration — maintenance

**Evidence:** [engine.py](/Users/devesh1011/code_with_devesh/projects/dromeus/src/dromeus/gossip/engine.py:251)
contains both `AXLPairTransport` and `GossipEngine`. The existing `PairTransport`
interface provides a seam for this extraction.

- [ ] Move AXL-specific bundle exchange, pair messages/retries, and related
  broadcasting into a gossip-owned adapter module. Keep generic round orchestration
  in the engine. Place shared interface/result types where imports stay acyclic.
- [ ] Keep transport tensor/codec blind and runtime responsible for composition.
  Avoid moving algorithm-aware bundle logic into the low-level transport package.
- [ ] Preserve readiness-before-transfer ordering, timeout/retry behavior, immutable
  exchange diagnostics, prepare/confirm durability, cancellation, and cleanup.
- [ ] Exercise the engine through `PairTransport` and the concrete adapter through
  protocol tests. Run readiness-skew, loss/duplicate/reorder/corruption, failure,
  durability, and formation regressions.

**Done when:** the engine is independent of AXL exchange implementation details,
protocol behavior is preserved, and structural checks show no dependency cycles.

## Verification and documentation for each implementation change

- [ ] Run focused behavioral checks, then `./scripts/verify` for the final change.
  Record environment-gated skips separately from executed real-AXL/NCCL evidence.
- [ ] Preserve the independent reference's optimizer/transport isolation.
- [ ] Update Obsidian current progress with implemented behavior and retained test
  evidence. Update Architecture and the relevant plan when decisions, interfaces,
  scope, or acceptance criteria change.
- [ ] Mark checklist items complete only after their stated acceptance checks pass.
