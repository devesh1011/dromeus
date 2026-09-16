# M2 NCCL reference evidence

This directory contains the compact NCCL comparison evidence intended for the
M2 technical report and Gensyn review. It retains final reports, per-rank logs,
run configuration, compact runtime metadata, checkpoint hash manifests, and
cleanup summaries.

Large local checkpoint binaries, evidence tarballs, dataset caches, raw AWS API
responses, provisioning attempts, and cancelled or failed runs were removed on
2026-09-10. Their removal does not change the recorded benchmark metrics. Where
checkpoint or archive hashes were part of an accepted run, those hashes remain
in the acceptance report or checkpoint manifest.

## Retained runs

| Scale | Seed | Status | Mean accuracy | Matching AXL | Evidence |
| --- | ---: | --- | ---: | ---: | --- |
| 4 nodes | 17 | accepted | 92.7100% | 93.0825% | `4-nodes/seed-17/` |
| 4 nodes | 29 | accepted | 92.9550% | 92.8500% | `4-nodes/seed-29/` |
| 8 nodes | 17 | accepted | 93.09125% | 93.07875% | `8-nodes/seed-17/` |
| 8 nodes | 29 | accepted | 92.85125% | 92.6525% | `8-nodes/seed-29/` |
| 16 nodes | 17 | accepted | 92.65125% | 92.928125% | `16-nodes/seed-17/` |
| 16 nodes | 29 | accepted | 92.48125% | 92.491875% | `16-nodes/seed-29/` |

Seeds 17 and 29 are accepted at all three scales. W4 seed 41 and W8 seed 41 were
cancelled before training; W16 seed 41 has not been run. The earlier W16 seed-29 host-maintenance failure is excluded from accepted
comparisons. Its separate failed-attempt package was recorded historically, but
was no longer present locally during commit preparation on 2026-09-16; it is not
included in this repository package. Its fresh rerun completed after maintenance controls
were added. The rerun retains a disclosed, caught rank-9 post-worker exit-barrier
warning, with all sixteen launchers exiting 0.

The new W16 seed-29 model-weight bytes remain in the original evidence checkout outside Git. The September
10 pruning record applies to the earlier packages.

## Review path

Start with each scale's `summary.json`, then open the referenced acceptance
report. Per-rank logs and summaries are retained beside each accepted report.
The frozen benchmark configuration is maintained once at
`../experiment-config/experiment.yaml`.

The NCCL and AXL deployments used different physical networks. The evidence
supports correctness and accuracy comparisons, not cross-fabric speed claims.
Final checkpoint manifests describe model weights only and do not claim
optimizer, loader, RNG, or resumable continuation state.

See `PRUNED-ARTIFACTS.md` for the local storage cleanup record.
