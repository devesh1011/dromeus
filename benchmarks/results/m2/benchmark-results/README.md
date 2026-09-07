# Dromeus M2 official evidence

Report-ready evidence for the frozen 45% top-k AXL runs. `4-nodes`, `8-nodes`, and `16-nodes`
mean world sizes of 4, 8, and 16 participating training nodes. The layout mirrors the
M1 `official/` archive while adding a scale level and keeping M2 reconciliation and
divergence artifacts explicit.

- `*-nodes/seed-*/logs/node-*/dromeus.jsonl`: complete per-node AXL/runtime JSONL logs.
- `*-nodes/seed-*/manifests/node-*/manifest.json`: sealed run manifests.
- `*-nodes/seed-*/run-metadata/node-*/`: state, topology snapshots, host exit code.
- `*-nodes/seed-*/analysis/node-*/analysis.jsonl`: durable checkpoint indexes; full temporary
  tensors were deleted after exact analysis.
- `*-nodes/seed-*/reports/`: acceptance, reconciliation, and divergence reports.
- `../experiment-config/`: experiment, pilot report, and deterministic initial checkpoints.
- `*-nodes/seed-*/launch/`: exact launch inputs used by each accepted run.
- `*-nodes/seed-*/final-checkpoints/node-*.json`: final checkpoint hashes, exact sizes,
  and versioned S3 archive locations. Large optimizer-state checkpoints are
  intentionally not duplicated locally.
- `*-nodes/checkpoint-checksums.sha256`: one portable SHA-256 manifest for the archived
  optimizer checkpoint objects.

Each accepted seed has one canonical report-ready location under this hierarchy.

Folder names describe the local layout. Recorded run IDs, selector profiles, and
versioned S3 object keys retain their original values. The hash-bound pilot report
is also retained at `../compression-calibration/pilot-report.json`; its
`evidence_root` is relative to that source directory.
