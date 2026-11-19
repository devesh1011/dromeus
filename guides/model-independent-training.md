# Task reference

Details of `dromeus.TrainingTask` beyond [training with Dromeus](application-training.md):
batches, losses, model checks, token data and trust.

## Fields

| Field | Meaning |
| --- | --- |
| `model_factory` | Builds the model. Must create only public initial state. |
| `batch_adapter` | A `TensorBatchAdapter` describing one batch's structure |
| `loss_per_record` | `(model, batch) -> Tensor[B]`, one differentiable loss per record |
| `model_definition`, `task_definition`, `loss_definition` | Text every node shares; hashed into the draft |
| `implementation_hash` | SHA-256 of the task code, usually `dromeus.code_hash(...)` |
| `optimizer` | `Optimizer.sgd/adam/adamw(...)` or an `OptimizerRegistration` |
| `batch_size` | Records per logical batch (the expected size under Poisson sampling) |
| `record_unit` | What one record is, for example `"example"` or `"fixed_token_sequence"` |
| `schedule` | `"constant"` or `"linear_decay"` learning rate |
| `initialization` | `"public_untrained"` or `"public_pretrained"` |
| `public_evaluate` | Optional `(model) -> {name: value}` on public data only |

## Batches

`TensorBatchAdapter(example, definition)` accepts dictionaries, tuples and lists of
dense tensors with records on axis 0, including integer token IDs and boolean
masks. Shapes after the record axis are fixed; pad sequences to a declared public
length. `example` is a public or synthetic batch of at least two records; it
describes the structure without touching any node's records. Collation must keep
each record intact and must not compute statistics across records. Constants that
aren't per record belong in the model or task code, not in the batch.

## Losses

Return a differentiable float tensor of shape `[B]`, one value per record. Don't
call `backward()` or step an optimizer inside the callback. Don't mix records,
read other records, keep raw activations across steps or publish raw losses.

## Token records

Language-model tasks fit the same interface: store fixed-length integer token
records, pad to a declared public length, and freeze the tokenizer vocabulary and
sequence boundaries before training. A per-sequence privacy statement does not
automatically protect a whole document or person; repeated or overlapping chunks
need composition.

## Model checks

Before any record is read, Dromeus checks the model on the public `example` batch:

- Supported: custom containers of compatible operations, shared (tied) parameters,
  declared frozen parameters and immutable public buffers.
- Each tied parameter is exchanged once. Frozen weights and public buffers are not
  exchanged; they are verified locally. Consensus for a partially frozen model
  measures trainable parameters only.
- Rejected, with the module path and the missing capability: BatchNorm,
  cross-record losses, unsafe trainable-layer buffers, uncontrolled model
  randomness, unbounded streams and changing record populations. There is no
  silent layer replacement or non-private fallback.

The private profile uses pinned Opacus hooks, dense float32 parameters, flat
clipping and deterministic task execution, and checks the example batch against
independent per-record autograd. This is evidence for that profile, not a proof
that arbitrary Python has no data-dependent side effects.

An empty Poisson draw still performs one noise-only update. Physical microbatches
(`LocalData.physical_batch_size`) do not add accounting steps.

## Checkpoints and exports

Local checkpoints contain optimizer, sampler and accountant state and must not be
published. Model-only exports keep the permitted model state, including aliases
and declared public constants, so they load into the same architecture. Secure
private runs cannot resume after a crash.

## Trust and identity

Implementation hashes and shared definitions show that peers agree; they do not
attest code remotely or sandbox a malicious local process. Install and inspect the
task code and its dependencies on every node. Dromeus never imports code named by
a peer's draft. Public initialization, public evaluation and no extra releases of
private data remain part of the privacy contract.
