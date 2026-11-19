# Example: custom regression

A complete, runnable task for four or more nodes.

| File | Purpose |
| --- | --- |
| [`custom_training.py`](custom_training.py) | The task: a three-feature network, Huber loss and Adam. `setup()` returns each node's `(task, LocalData)`. |
| [`custom-regression-draft.yaml`](custom-regression-draft.yaml) | The shared draft, written by `dromeus draft` (command below). |
| [`node.yaml`](node.yaml) | One node's configuration. Copy and adapt it for every machine. |

Each node supplies its own NPZ file with float32 `inputs` `[N, 3]` and `targets`
`[N, 1]`. `N` may differ between nodes. Evaluation uses a small fixed synthetic
holdout defined in the example, never a node's records.

Before running:

1. Start an AXL node on each machine and connect the group's peers.
2. On one machine, write a fresh draft with your AXL version and give the same file
   to every node:

   ```bash
   uv run dromeus draft examples/custom-regression-draft.yaml \
     --task examples.custom_training:task --run-id custom-regression \
     --participants 4 --rounds 20 --local-steps 10 --axl-version YOUR-AXL-VERSION
   ```

   The committed draft records the Dromeus and PyTorch versions it
   was written with; each node must have those versions installed.
3. Adapt `node.yaml` per machine: set one node's `role` to `initiator`, the rest to
   `participant`, and deliver the initiator's invitation file to each
   participant's `invitation_path`.

Then on each node, from the repository root:

```bash
DROMEUS_LOCAL_DATA=/absolute/path/to/this-nodes-data.npz \
  uv run dromeus run --config examples/node.yaml \
  --task examples.custom_training:setup
```

The [training guide](../guides/application-training.md) explains the interface
this example uses, and the [configuration guide](../guides/configuration.md) explains
the node configuration and draft.
