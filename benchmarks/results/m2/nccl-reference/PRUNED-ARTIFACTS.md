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
