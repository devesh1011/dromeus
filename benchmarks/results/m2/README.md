# Dromeus M2 benchmark evidence

This directory contains the experiment configuration, compression calibration, and
accepted results for NoLoCo training over AXL. All nine compressed AXL runs at
4, 8, and 16 nodes, using seeds 17, 29, and 41, are accepted.

The paper-length NCCL comparisons, four-node identity-codec accuracy ablation,
consolidated technical report, and `v0.2.0` release remain open. The JSON reports
here are supporting evidence for that submission.

## Where to start

| Directory | Purpose |
|---|---|
| [benchmark-results/](benchmark-results/) | Accepted runs, per-run reports, logs, manifests, and checkpoint archive records |
| [experiment-config/](experiment-config/) | The exact shared experiment, hash-bound pilot report, and initial model checkpoints |
| [compression-calibration/](compression-calibration/) | The four-node experiment used to validate the 45% top-k compression setting before the official runs |

AWS deployment inputs remain outside this evidence directory in
[aws/results/orchestration/](../../../aws/results/orchestration/). Its
[matrix.json](../../../aws/results/orchestration/matrix.json) indexes all nine
accepted runs. Each accepted run also retains its actual launch inputs in its
own `launch/` directory.

## Accepted benchmark runs

Each cell links to that run's acceptance report. A seed identifies a reproducible
experiment repetition; a node/rank identifies one participating worker.

| Participating nodes | Seed 17 | Seed 29 | Seed 41 |
|---|---|---|---|
| 4 | [Accepted](benchmark-results/4-nodes/seed-17/reports/acceptance-report.json) | [Accepted](benchmark-results/4-nodes/seed-29/reports/acceptance-report.json) | [Accepted](benchmark-results/4-nodes/seed-41/reports/acceptance-report.json) |
| 8 | [Accepted](benchmark-results/8-nodes/seed-17/reports/acceptance-report.json) | [Accepted](benchmark-results/8-nodes/seed-29/reports/acceptance-report.json) | [Accepted](benchmark-results/8-nodes/seed-41/reports/acceptance-report.json) |
| 16 | [Accepted](benchmark-results/16-nodes/seed-17/reports/acceptance-report.json) | [Accepted](benchmark-results/16-nodes/seed-29/reports/acceptance-report.json) | [Accepted](benchmark-results/16-nodes/seed-41/reports/acceptance-report.json) |

These runs use GroupNorm ResNet-18 on disjoint IID CIFAR-10 partitions, with 500
outer rounds of 50 local steps: 25,000 inner steps per node. The frozen compression
uses bitmap top-k int8 at 45% for outer gradients and dense int8 for slow weights.
Exact optimizer, schedule, data, membership, hardware, and evidence settings are
recorded in [experiment.yaml](experiment-config/experiment.yaml).

## Why calibration has only four ranks

[compression-calibration/](compression-calibration/) records one four-node,
seed-17 calibration run. It checked accuracy, error-feedback residual bounds,
and communication reduction for the proposed 45% top-k setting. Its
[pilot-report.json](compression-calibration/pilot-report.json) records the outcome.

The validated settings were then used for the nine runs in `benchmark-results/`.
The `axl/` and `evidence/` folders inside calibration therefore contain four
workers; the eight- and sixteen-node results belong in `benchmark-results/`.

## Checkpoints and supporting evidence

- `experiment-config/checkpoint-17.safetensors`, `checkpoint-29.safetensors`, and
  `checkpoint-41.safetensors` are initial, untrained model weights. All scales for
  a given seed use the same starting weights. The frozen experiment also includes
  a separate four-node trajectory selector.
- `final-checkpoints/node-*.json` files inside accepted runs identify the final
  optimizer checkpoints archived in versioned S3. They contain hashes, sizes, and
  object locations; access to the private archive is separate from this repository.
- `analysis/node-*/analysis.jsonl` files are snapshot indexes. Temporary full
  analysis tensors were deleted after divergence analysis; the resulting reports
  and hashes remain.
- `reports/` contains acceptance, divergence, and communication-reconciliation
  evidence. `logs/`, `manifests/`, and `run-metadata/` retain the underlying records.
  See the [evidence layout](benchmark-results/README.md) for details.

## Reproduction and integrity

The local baseline commit includes source, frozen configuration, compact reports,
one sealed manifest per accepted run, and checkpoint archive records. Raw logs,
initial checkpoint binaries, full reconciliation output, and remaining per-node
records are retained locally outside Git. Their hashes remain in the frozen
experiment and package checksum manifests. Downloadable evidence archives have not
yet been published; a clean checkout alone does not include these external files.

Use the [Dromeus runner](../../noloco/dromeus_official_runner.py),
[independent reference runner](../../noloco_reference/runner.py), and
[pinned runtime record](../../noloco/container/runtime-lock.json) with the frozen
experiment. Recorded launch files include historical machine addresses and paths;
adapt deployment addresses for your environment while preserving the intended
experiment settings. Run IDs and S3 object keys retain their original names.

Package `checksums.sha256` files use paths relative to their containing directory.
For example, from the repository root:

```sh
cd benchmarks/results/m2/benchmark-results/4-nodes
shasum -a 256 -c checksums.sha256
```

Repeat for `8-nodes/` and each of the three `16-nodes/seed-*/` directories. These
checks cover the complete evidence packages, including raw logs. If large files
are distributed separately, restore them to their original relative paths first.
The `checkpoint-checksums.sha256` files describe archived optimizer objects and
require those objects to be retrieved separately.

The frozen loader validates the experiment's model, pilot report, initial
checkpoints, partitions, seeds, and pairing digests. Keep hash-bound input files
unchanged. The pilot report copied into `experiment-config/` retains the same
bytes as its calibration copy; its `evidence_root` is relative to
`compression-calibration/`. Historical paths inside measured evidence are
provenance, not current local download locations.

These results establish behavior on the declared IID workload. They do not
establish non-IID learning quality or complete the outstanding NCCL comparisons.
