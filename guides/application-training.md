# Training with Dromeus

Every node in a group trains the same model on its own records. After each round
of local steps, a node exchanges a NoLoCo update with one scheduled peer over AXL.
No node ever sends its records anywhere.

You write three things:

1. A **task**: the model, how records form a batch, the per-record loss and the
   optimizer. Every node builds the identical task.
2. A **draft**: the group's shared settings (members, rounds, NoLoCo settings).
   One person writes it from the task, and every node uses the same file.
3. Each node's **local data**: its own records, which never leave the machine.

The [example](../examples/custom_training.py) is a complete regression task in
about 120 lines. Everything below imports only from `dromeus`.

## Define the task

```python
from pathlib import Path

import torch
from torch import nn

import dromeus


def build_model() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 7), nn.Tanh(), nn.Linear(7, 1))


def huber_per_record(model, batch):
    inputs, targets = batch
    losses = nn.functional.huber_loss(model(inputs), targets, reduction="none")
    return losses.mean(dim=1)  # one loss per record, shape [B]


def task() -> dromeus.TrainingTask:
    return dromeus.TrainingTask(
        model_factory=build_model,
        batch_adapter=dromeus.TensorBatchAdapter(
            example=(torch.zeros(2, 3), torch.zeros(2, 1)),  # public, at least 2 records
            definition="inputs [B,3] float32, targets [B,1] float32",
        ),
        loss_per_record=huber_per_record,
        model_definition="regression: Linear(3,7), Tanh, Linear(7,1); float32",
        task_definition="regression: three float32 features; scalar float32 target",
        loss_definition="Huber loss, delta 1, one value per record",
        implementation_hash=dromeus.code_hash(Path(__file__)),
        optimizer=dromeus.Optimizer.adam(learning_rate=0.005),
        batch_size=8,
    )
```

- The **definitions** are short descriptions of what the code means. They become
  hashes in the draft, so every node must use identical text. Change the text
  whenever you change the meaning.
- `implementation_hash` identifies the code itself. `code_hash` hashes the files
  you name, so pass every module your task depends on.
- `loss_per_record` computes the forward pass and returns one differentiable loss
  per record. Don't call `backward()` or step an optimizer: Dromeus does both.
- The optimizer is `Optimizer.sgd`, `Optimizer.adam` or `Optimizer.adamw`. For
  any other optimizer, see [private training](private-training.md#register-an-optimizer).
- Optional `public_evaluate(model) -> {name: value}` reports metrics each round.
  It must use public data, never a node's own records.

## Write the shared draft

One person writes the draft once and gives the file to every node:

```python
dromeus.make_draft(
    task(),
    "draft.yaml",
    run_id="regression-2026-11",
    participants=4,
    rounds=20,
    local_steps=10,
)
```

`make_draft` fills in every hash, plus the Dromeus and PyTorch versions installed
on this machine and the AXL build bundled with Dromeus. Every node must install
those same versions. Where no AXL is bundled, pass `axl_version` naming the AXL
build every node runs. Optional settings:
`noloco=dromeus.NoLoCo(alpha=..., beta=..., gamma=..., top_k_fraction=...)` for the
outer step and update compression, `privacy=dromeus.Privacy(...)` for private
training, `transport=dromeus.TransportLimits(...)` for WAN tuning and
`divergence=dromeus.DivergencePolicy(...)` for [divergence warnings](divergence-monitoring.md).

The same draft, without the optional settings, from the command line:

```bash
dromeus draft draft.yaml --task my_training:task --run-id regression-2026-11 \
  --participants 4 --rounds 20 --local-steps 10
```

Each node can check the draft before starting:

```python
dromeus.check_draft("draft.yaml", task())  # raises DraftError naming what differs
```

or `dromeus check --draft draft.yaml --task my_training:task`.

## Run each node

Each node has its own [node configuration](configuration.md) naming its AXL bridge,
its role and where to write files. Then:

```python
import asyncio

local = dromeus.LocalData(
    dataset=my_records,          # any indexable torch Dataset of record tensors
    data_identity=my_file_hash,  # names this exact copy of the records
)
node = dromeus.Node.from_file("node.yaml")
result = asyncio.run(node.run(task(), local=local))
print(result.committed_rounds, result.manifest_hash)
```

Or from the command line, with a function that returns `(task, local_data)`:

```bash
dromeus run --config node.yaml --task my_training:setup
```

See the [command-line guide](command-line.md) for `doctor`, `watch` and exit codes.

Before contacting AXL, the node checks the draft against its task and its
installed Dromeus and PyTorch versions. One node is the `initiator`: it writes an
invitation file, which you deliver to every participant at its configured
`invitation_path`. The group forms, trains every round and stops. `run` returns
when the run completes, or raises a `DromeusError` subclass.

## What happens in a round

Your optimizer performs `local_steps` steps on the node's own batches. Dromeus then
computes NoLoCo's outer gradient from the fast and slow weights, exchanges the
outer gradient and slow weights with this round's peer, and applies the outer
momentum and correction. Each committed round is recorded as a `round_metrics`
evidence record with the training loss and any public evaluation metrics.
Checkpoints are written under the node's `run_root`.

## Errors

Every failure Dromeus raises on purpose is a `dromeus.DromeusError`:

| Error | Meaning |
| --- | --- |
| `ConfigError` | The node configuration or a local setting is invalid |
| `TaskError` | The task's model, batches, loss or optimizer is not admissible |
| `DraftError` | The draft is malformed or doesn't match this task or machine |
| `PrivacyError` | A privacy policy, ledger or budget check failed |
| `FormationError` | The group couldn't form: invitation, membership or readiness |
| `RunError` | The run failed after formation: peers, transport or rounds |

The first four are also `ValueError`; the last two are also `RuntimeError`.

## Limits

- Model parameters must be dense float32. Represent a scalar parameter as shape
  `[1]`. Integer buffers stay local and are checkpointed.
- Every node weighs equally, whatever its number of records.
- Membership is fixed: if one node fails, the run stops. There is no automatic
  group restart.
- Batches are trees (tuples, lists, string-keyed dicts) of tensors with records on
  axis 0 and fixed trailing shapes. Pad sequences to a declared length.

For token data, frozen weights and the exact model checks, see the
[task reference](model-independent-training.md).
