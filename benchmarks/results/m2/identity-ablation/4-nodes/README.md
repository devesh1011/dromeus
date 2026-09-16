# W4 seed-17 identity-codec ablation

This is the accepted four-node AXL identity-codec isolation run:
`noloco-identity-official-w4-s17-retry2-20260910`.

- Four `g5.xlarge` NVIDIA A10G workers, two in `us-east-1` and two in `eu-north-1`.
- Frozen 500 outer rounds and 25,000 local Adam steps per worker.
- `identity-v1` for both `outer_gradient` and `slow_weights`.
- Mean final accuracy: 92.71%.
- Matching compressed AXL W4 seed-17 mean: 93.0825%, a difference of -0.3725 percentage points.
- Logical identity update: 89,402,768 bytes. Complete wire update: 89,438,360 bytes before one observed metric-level retry.
- Strict reconciliation, divergence, checkpoint, runtime identity, and cleanup gates passed.

Start with [reports/acceptance-report.json](reports/acceptance-report.json).
The historical [checksums.sha256](checksums.sha256) covers the original retained
package, including model-weight binaries now excluded from Git. The repository-wide
`package-checksums.sha256` verifies the files shipped in this checkout.
The original raw capture was pruned locally after validation. Its hashes and
sizes remain recorded in the reports; the retained report-facing logs and metadata are committed here. Model-weight
binaries remain outside Git in the original evidence checkout.
