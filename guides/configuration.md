# Application configuration

Each participant needs two files: a node configuration for its own machine, and
the shared run draft that every participant must agree on byte for byte. Runnable
copies live in [`examples/`](../examples/): [`node.yaml`](../examples/node.yaml)
and [`custom-regression-draft.yaml`](../examples/custom-regression-draft.yaml),
which match the [example application](../examples/custom_training.py).

Replace environment versions, definition hashes, paths, bridge addresses and
participant roles with your application's values.

## Node configuration

```yaml
# Adapt paths and bridge URL independently on each machine.
role: initiator
draft_path: /path/to/custom-regression-draft.yaml
axl_bridge_url: http://127.0.0.1:9002
run_root: runs/custom-regression/node-0
invitation_path: runs/custom-regression/invitation.json
bootstrap_uri: axl://custom-regression-v1
invitation_timeout_seconds: 300
```

## Shared draft

```yaml
# Replace environment versions and commit before starting the group.
manifest_version: 7
run_id: custom-regression-v1
algorithm_id: noloco
model_id: custom-regression
model_definition_hash: 31df3d1a8ba81fa6d1245d7db5da9ddaff011c3b2b0b0ab6dfbe9861e71ed892
dataset:
  dataset_id: application
  task_id: regression
  definition_hash: f3816a5af54d8906f730393ab5f8e619de7edda81e75a0dc849092585eee6a1a
application_training:
  policy_id: rmsprop-huber
  definition_hash: 1a983898d8d0cda650ab854a193edacf20c310037cbeb2aa06726742f7945e70
expected_participant_count: 4
environment:
  dromeus_version: 0.1.0
  dromeus_commit: '0000000'
  pytorch_version: replace-with-installed-version
  axl_version: replace-with-installed-version
  model_definition_hash: 31df3d1a8ba81fa6d1245d7db5da9ddaff011c3b2b0b0ab6dfbe9861e71ed892
local_steps: 10
round_count: 20
peer_scheduler_seed: 17
codec_id: safetensors
algorithm_config:
  alpha: 0.5
  beta: 0.7
  gamma: 0.7
  inner_steps: 10
artifact_codecs:
- artifact_name: outer_gradient
  codec_id: identity
- artifact_name: slow_weights
  codec_id: identity
transport:
  max_payload_bytes: 8388608
  max_retries: 3
  retry_timeout_seconds: 5.0
  chunk_size_bytes: 65536
  window_size: 2
consensus_sketch:
  size: 4096
  seed: 17
```

The draft uses manifest version 7, the only version this release accepts. The
three definition hashes are SHA-256 digests of your application's model, task and
training definitions (`dromeus.application.definition_hash`); every participant
must compute the same values. `local_steps` must equal
`algorithm_config.inner_steps`.

`artifact_codecs` selects how updates are encoded on the wire. The two supported
combinations are `identity` for both artifacts, or `topk-bitmap-int8` (with
`top_k_fraction` and `lossy_allowed: true`) for `outer_gradient` together with
`dense-int8` (with `lossy_allowed: true`) for `slow_weights`.

Private and model-independent task runs add `privacy` and `private_training`
sections; see [private training](private-training.md) and
[model-independent training](model-independent-training.md).
