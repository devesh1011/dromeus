# W16 seed 29 NCCL reference

Run `noloco-nccl-w16-s29-r2-20260914` is accepted as final-model-weight evidence.
All sixteen A10G workers completed 500 outer rounds and 25,000 local Adam steps,
with exit code 0. Mean final accuracy is 92.48125%, versus 92.491875% for matching
AXL seed 29. The AXL-minus-NCCL difference is +0.010625 percentage points.

Start with [the acceptance report](reports/acceptance-report.json).
Elapsed run time was 4,475 seconds, from 14:39:54 to 15:54:29 UTC on 2026-09-14.
All task-owned AWS resources were removed after capture and validation.

Rank 9 logged a caught torchrun exit-barrier network exception after writing its
final checkpoint and summary. Its launcher still exited 0. The original log,
artifact timestamps, and [warning analysis](reports/launcher-warnings.json) are
retained; this warning is not silently treated as absent.

Sixteen final safetensors files (715,226,624 bytes total) remain in the original
evidence checkout's `checkpoints/` directory, outside Git. Each passed remote and local finite-FP32/schema validation and exact
SHA-256 matching. These are model weights, not resumable optimizer/loader/RNG
state. The checksum manifest covers the full retained local package; access to
ignored files is necessary to verify every entry after checkout elsewhere.

The earlier host-maintenance failure is excluded from accepted metrics. Its
separate `../failed-seed-29/` capture is not present in the current local evidence
checkout or this Git package; earlier retention statements are historical. No normalized cross-fabric
speed comparison is claimed.
