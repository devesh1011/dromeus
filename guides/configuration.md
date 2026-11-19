# Configuration

Each node needs two files:

- its own **node configuration**, which only that machine uses;
- the group's shared **draft**, written once by `dromeus.make_draft` and copied
  unchanged to every node.

Runnable copies are in [`examples/`](../examples/): [`node.yaml`](../examples/node.yaml)
and [`custom-regression-draft.yaml`](../examples/custom-regression-draft.yaml),
generated from the [example task](../examples/custom_training.py).

## Node configuration

```yaml
role: initiator                         # exactly one node; the others: participant
draft_path: /path/to/draft.yaml
axl_bridge_url: http://127.0.0.1:9002   # this machine's AXL bridge, loopback only
run_root: runs/regression/node-0        # checkpoints, logs and run state
invitation_path: runs/regression/invitation.json
bootstrap_uri: axl://regression-2026-11 # the same on every node
invitation_timeout_seconds: 300        # participants: how long to wait for the invitation
enrollment_window_seconds: 900          # initiator only, optional: when joining closes
```

The initiator writes the invitation to `invitation_path`. Copy that file to the
same path on every participant; participants wait up to
`invitation_timeout_seconds` for it. The AXL bridge must already be running and
connected to the group's peers. `dromeus doctor --config node.yaml` checks all of
this before you start.

With `enrollment_window_seconds`, the invitation expires that many seconds after
the initiator writes it. Participants refuse an expired invitation, and the
initiator stops with a `FormationError` if the group isn't complete when the window
closes, so a stale invitation can't start a run late. Expiry compares wall clocks
across machines: keep every node's clock NTP-synchronized. Without the setting,
invitations don't expire and formation is bounded only by its control deadline.

## Shared draft

Write the draft with `dromeus.make_draft` (see
[training with Dromeus](application-training.md#write-the-shared-draft)). Don't
edit the hashes by hand: change the task and write a new draft. The arguments
map to the draft like this:

| `make_draft` argument | Draft field | Notes |
| --- | --- | --- |
| `run_id` | `run_id` | Use a new ID for every run |
| `participants` | `expected_participant_count` | Even, 4 to 16 |
| `rounds` | `round_count` | |
| `local_steps` | `local_steps`, `algorithm_config.inner_steps` | Optimizer steps per round |
| `axl_version` | `environment.axl_version` | The AXL build the group pins |
| `noloco` | `algorithm_config`, `artifact_codecs` | `top_k_fraction` turns on int8 compression |
| `privacy` | `privacy`, `training.sampling` | Private runs use Poisson sampling |
| `transport` | `transport` | Retry, chunk and window limits for the network |
| `divergence` | `divergence_detection` | Optional warnings |
| `seed` | `peer_scheduler_seed`, `consensus_sketch.seed` | Peer schedule order |

The `training` block records the task's optimizer, batch size, sampling and the
hashes of its definitions and code. `environment` records the Dromeus and PyTorch
versions of the machine that wrote the draft; every node must install the same
versions, and a node refuses a draft written for different ones. PyTorch is
matched by release, so `2.13.0` on macOS, `2.13.0+cpu` on Linux and `2.13.0+cu130`
on a CUDA machine can share one group; each node logs its exact build in its
`node_start` event.

The draft uses manifest version 7, the only version this release accepts.
