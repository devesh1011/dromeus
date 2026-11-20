# Changelog

From 0.4.0, the names exported from `dromeus` are the public API. They change
only in a minor release, with an entry here. Everything else (modules under
`dromeus.*`, the event log format, the command-line output) may change in any
release. Until 0.4.0 itself, pre-releases may still change the public API.

Releases before 0.4.0 were research milestones (M1 to M3), kept at their tags
`v0.1.0`, `v0.2.0` and `v0.3.0`. They have no entries here, and 0.4.0 doesn't read
their formats.

## 0.4.0rc1 (2026-11-20)

The first release meant for use outside the project: a library with a small
command line, installable from PyPI, with a bundled AXL node.

### Added

- Public library API exported from `dromeus`: `TrainingTask`, `make_draft`,
  `check_draft`, `Node`, `LocalData`, the optimizer, privacy, NoLoCo, transport
  and divergence settings, and the `DromeusError` hierarchy (`ConfigError`,
  `TaskError`, `DraftError`, `PrivacyError`, `FormationError`, `RunError`).
- Command line: `dromeus draft`, `check`, `doctor`, `run` and `watch`, with
  documented exit codes.
- `dromeus doctor` checks a node before it starts: configuration, draft, task,
  AXL build, the AXL bridge, its key, peer links, and the clock.
- `dromeus watch` follows a node's rounds, loss and consensus distance live.
- The pinned AXL node is installed with Dromeus on Linux (x86_64, aarch64) and
  Apple-silicon macOS, as the `dromeus-axl` package and command. Drafts record its
  build, and `doctor` warns when a draft names another.
- `enrollment_window_seconds`: invitations expire, and the initiator stops if the
  group isn't complete in time.
- Python 3.11, 3.13 and 3.14 support, alongside 3.12.
- A four-node quickstart on one machine (`guides/quickstart.md`), run as written
  by the test suite, and an AXL setup guide (`guides/axl.md`).

### Changed

- Training is defined by a task (model, batches, per-record loss, optimizer) and
  a shared draft, with manifest version 7 the only accepted format.
- Drafts pin PyTorch by release, not build: `2.13.0`, `2.13.0+cpu` and
  `2.13.0+cu130` nodes can train together. Each node logs its exact build.

### Fixed

- A node that committed every round no longer fails when its end-of-run AXL
  topology snapshot can't be taken; the failure is logged as
  `topology_snapshot_failed`.
- `dromeus watch` no longer waits forever for a node that stopped without logging
  a result. A running node holds `run_root/node.lock`, which also keeps two nodes
  from sharing one run root.
- Participants wait for a completely copied invitation file instead of failing
  on a partial one.
- A malformed draft is an input error (exit 3), not an unexpected one.

### Removed

- D-PSGD and the legacy codec paths. NoLoCo is the only algorithm.
