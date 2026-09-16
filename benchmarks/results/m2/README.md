# Dromeus M2 benchmark evidence

## Start here

- [Reviewed technical report (PDF)](report/M2_Gensyn_Submission_Report.pdf)
- [Working Google Doc](https://docs.google.com/document/d/10zjwEh-Yp9df0_d9XhJQrnXjkPerIIuq9dKUAuXVBCM/edit)
- [Per-run accuracy table (CSV)](report/metrics.csv) and [metrics, comparisons and source hashes (JSON)](report/metrics.json)
- [Four figures](report/charts/) and [compact chart data](report/chart-data.json)
- [Accepted-run index](run-matrix.json) and [artifact availability](evidence-availability.json)

Nine compressed AXL runs, six NCCL references, and one identity-codec AXL ablation
are accepted as retained benchmark evidence. This records internal validation,
not final Gensyn acceptance. The `v0.2.0` release remains unpublished.

## Accepted runs

The AXL runs use the frozen 45% bitmap top-k/int8 configuration. Every accepted
run completes 500 outer rounds and 25,000 local steps per worker. References
share the frozen workload, initialization, schedule, and accelerator class;
regional topology and timing boundaries differ.

| Workers | AXL seed 17 | AXL seed 29 | AXL seed 41 | NCCL seed 17 | NCCL seed 29 |
|---|---|---|---|---|---|
| 4 | [17](benchmark-results/4-nodes/seed-17/reports/acceptance-report.json) | [29](benchmark-results/4-nodes/seed-29/reports/acceptance-report.json) | [41](benchmark-results/4-nodes/seed-41/reports/acceptance-report.json) | [17](nccl-reference/4-nodes/seed-17/acceptance-report.json) | [29](nccl-reference/4-nodes/seed-29/reports/acceptance-report.json) |
| 8 | [17](benchmark-results/8-nodes/seed-17/reports/acceptance-report.json) | [29](benchmark-results/8-nodes/seed-29/reports/acceptance-report.json) | [41](benchmark-results/8-nodes/seed-41/reports/acceptance-report.json) | [17](nccl-reference/8-nodes/seed-17/reports/acceptance-report.json) | [29](nccl-reference/8-nodes/seed-29/reports/acceptance-report.json) |
| 16 | [17](benchmark-results/16-nodes/seed-17/reports/acceptance-report.json) | [29](benchmark-results/16-nodes/seed-29/reports/acceptance-report.json) | [41](benchmark-results/16-nodes/seed-41/reports/acceptance-report.json) | [17](nccl-reference/16-nodes/seed-17/reports/acceptance-report.json) | [29](nccl-reference/16-nodes/seed-29/reports/acceptance-report.json) |

The [four-worker seed-17 identity ablation](identity-ablation/4-nodes/reports/acceptance-report.json)
provides the uncompressed quality control. No NCCL seed-41 result is imputed.
Per-worker population SD and across-seed sample SD are kept separate in the
report and machine-readable metrics.

## What is in Git

- [Compressed AXL packages](benchmark-results/): accepted-run logs, manifests,
  launch inputs, analysis indexes, validation/reconciliation/divergence reports,
  summaries, and versioned checkpoint references.
- [NCCL packages](nccl-reference/): per-rank logs and summaries, acceptance and
  validation reports, runtime/source provenance, capture wrappers, and checksums.
- [Identity package](identity-ablation/4-nodes/): AXL/Dromeus logs, manifests,
  summaries, validation, divergence, and checkpoint hash records.
- [Frozen experiment](experiment-config/experiment.yaml) and hash-bound pilot
  report. [Compression calibration](compression-calibration/) retains the
  pre-freeze 45% run and its logs; it is distinct from official repetitions.

The executable benchmark and reference remain under `benchmarks/noloco/` and
`benchmarks/noloco_reference/`. Recorded launch addresses and filesystem paths
are historical deployment provenance, not current endpoints.

## External and unavailable artifacts

Model/checkpoint binaries, dataset caches, raw provisioning captures, and
credentials are excluded from Git. The [availability inventory](evidence-availability.json)
records sizes and SHA-256 values for 23 model files retained in the original
local evidence checkout. No public download URL is claimed for those files.
AXL final-optimizer references are under each run's `final-checkpoints/node-*.json`;
private S3 access must be arranged and the bytes verified separately.

Earlier NCCL checkpoint binaries and some raw archives were pruned locally;
see the [pruning ledger](nccl-reference/PRUNED-ARTIFACTS.md). The earlier failed
W16 NCCL seed-29 attempt remains excluded from metrics, but its separate failure
capture was absent during this commit preparation. Historical retention claims
in original reports do not establish current availability. Accepted
W16 seed-29 logs retain the post-worker warning and its analysis.

The PDF is the reviewed 17 September revision. Appendix B links the published
review package and distinguishes external and missing artifacts. No run metric
or historical acceptance report was rewritten to hide a failure or missing artifact.

## Verify and reproduce

From this directory, verify the committed package:

```sh
shasum -a 256 -c package-checksums.sha256
```

This manifest covers files shipped in Git, excluding itself. Original per-run
`checksums.sha256` files are preserved as historical full-package manifests;
some also reference external or pruned files and cannot all pass in a Git-only
checkout. A recorded hash is not a downloadable copy.

See [report reproduction](report/README.md) for metrics and chart commands.
Re-running training also requires the hash-bound initial checkpoint bytes,
CIFAR-10 data, pinned GPU runtime, and newly provisioned deployment addresses.
Keep frozen experiment/pilot bytes unchanged. The four-node calibration is
not an additional independent official seed.

Results cover an IID CIFAR-10 vision workload. They do not establish general
non-IID or LLM convergence, pipeline parallelism, or automatic distributed restart.
