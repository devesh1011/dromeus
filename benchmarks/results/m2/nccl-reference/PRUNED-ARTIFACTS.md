# NCCL evidence storage cleanup

On 2026-09-10, the NCCL evidence tree was reduced from 4,675 files and
7,496,028,160 allocated bytes to a compact report-facing package.

The cleanup removed these local categories:

- final model checkpoint binaries after their SHA-256 values, tensor schema,
  byte sizes, and validation results had been retained in reports;
- duplicate frozen input checkpoints;
- final evidence tarballs and archive listings;
- CIFAR caches and temporary trajectory snapshots;
- raw AWS instance, volume, peering, subnet, route, and security-group dumps;
- capacity probes, staging transcripts, image-pull logs, and retry debris;
- cancelled and capacity-blocked seed attempts;
- duplicate checkpoint-capture preflight packages.

Important removed archive identities remain recorded for audit:

| Run | Original archive size | SHA-256 |
| --- | ---: | --- |
| W4 seed 17 | 580,383,926 bytes | `e5bd9dd24c4e4ac82fa48648a6344f8c5ae800ab8dd0cb4f0e3b0863638d5a8d` |
| W8 seed 29 | 456,128,987 bytes | `5afefad736fc11a78125f2b659b4e2c2c006ee849cd5fbdc4fe84f56e09fc544` |
| W16 seed 17 | 953,888,756 bytes | `ef32413e7301090f07e53026dda9bd90d4df048c360bf69abc3d37253756fb5a` |

These archive bytes are no longer present locally. If Gensyn requests complete
binary evidence, a new versioned external artifact location must be prepared
and its URL, object version, size, and SHA-256 added to the technical report.
The current grant-facing package retains the requested per-node logs and run
manifests.

## 2026-09-16: W8 seed-17 summary deduplication

Removed eight byte-identical nested copies after comparing each with the
retained flat per-rank summary. Accuracy, loss, completed-step results, logs,
run metadata, and capture/validation wrappers are unchanged.

| Removed duplicate | Retained canonical file | SHA-256 |
|---|---|---|
| `8-nodes/seed-17/summaries/rank-0/rank-0/summary.json` | `8-nodes/seed-17/summaries/rank-0-summary.json` | `1be2143a50702801cb8c5c160145662c127650012e5529a597a64bd948ae150c` |
| `8-nodes/seed-17/summaries/rank-1/rank-1/summary.json` | `8-nodes/seed-17/summaries/rank-1-summary.json` | `03da32ad7e10cc74ae84487a76639533835854d61a64a44a171b92de549fd94a` |
| `8-nodes/seed-17/summaries/rank-2/rank-2/summary.json` | `8-nodes/seed-17/summaries/rank-2-summary.json` | `d7be67a605e1083a1513d77979a97f4a81cb3e15d43497d2f0d4931051a99ef1` |
| `8-nodes/seed-17/summaries/rank-3/rank-3/summary.json` | `8-nodes/seed-17/summaries/rank-3-summary.json` | `28ec0cffddd641f0e829bde1116a7dfd8c9d7790a1685c8e43a5ce65c7fd2c7f` |
| `8-nodes/seed-17/summaries/rank-4/rank-4/summary.json` | `8-nodes/seed-17/summaries/rank-4-summary.json` | `2e2cd1ada6533db5d1ee3fe1ccc46e8ef97965e8674b6aff7a06bd184d871708` |
| `8-nodes/seed-17/summaries/rank-5/rank-5/summary.json` | `8-nodes/seed-17/summaries/rank-5-summary.json` | `9fa40e76f393ee7b5cb62c5855801f535d849b8d0fe55a74802571641781313c` |
| `8-nodes/seed-17/summaries/rank-6/rank-6/summary.json` | `8-nodes/seed-17/summaries/rank-6-summary.json` | `0f0ecfdccfb20b7ca448fdcb5fd4d141652823620d37162c512de6d0f575e28b` |
| `8-nodes/seed-17/summaries/rank-7/rank-7/summary.json` | `8-nodes/seed-17/summaries/rank-7-summary.json` | `5e598a6d183dcf63892e1ff52619073d0a92d04ffb55d8fa2570c80574640993` |

The original per-run and scale `checksums.sha256` files remain historical
full-package records and still list the original duplicate paths. Their hashes
are preserved. Use `../package-checksums.sha256` from this directory for the
current Git package, with paths resolved from `benchmarks/results/m2/`. The table
above maps each historical duplicate to identical retained bytes.
