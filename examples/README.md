# Example: custom regression application

A complete, runnable application for four or more participants.

| File | Purpose |
| --- | --- |
| [`custom_training.py`](custom_training.py) | The application: a three-feature network, Huber loss and RMSprop. `prepare_training` is the factory each node runs. |
| [`custom-regression-draft.yaml`](custom-regression-draft.yaml) | The shared run draft (manifest version 7). Its definition hashes match `custom_training.py`. |
| [`node.yaml`](node.yaml) | One node's configuration. Copy and adapt it for every machine. |

Each node supplies its own NPZ file with FP32 `inputs` `[N, 3]` and `targets`
`[N, 1]`, plus optional held-out `evaluation_inputs` and `evaluation_targets`.
`N` may differ between nodes.

Before running:

1. Start an AXL node on each machine and connect the group's peers.
2. Replace the `environment` versions in the draft with your deployment's values;
   every participant must use the identical draft.
3. Adapt `node.yaml` per machine: set one node's `role` to `initiator`, the rest to
   `participant`, and deliver the initiator's invitation file to each
   participant's `invitation_path`.

Then on each node, from the repository root:

```bash
DROMEUS_LOCAL_DATA=/absolute/path/to/this-nodes-data.npz \
  uv run python -m dromeus.node --config examples/node.yaml \
  --factory examples.custom_training:prepare_training
```

The [application training guide](../guides/application-training.md) explains the
interface this example uses, and the [configuration guide](../guides/configuration.md)
explains every draft field. Changing the model, task or training policy requires
updating the matching definitions and hashes in the draft.
