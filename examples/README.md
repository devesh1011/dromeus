# Application-owned training

Your Python program owns the model, data loader, loss, optimizer, scheduling, and
local evaluation. Dromeus forms the group, validates the shared contract and tensor
schema, runs NoLoCo peer updates over AXL by default, and persists checkpoints.

[`custom_training.py`](custom_training.py) is a regression example with a custom
three-feature network, Huber loss, and RMSprop. Each node supplies its own local
NPZ file. The runtime never selects a model or downloads a dataset.

```bash
DROMEUS_LOCAL_DATA=/absolute/path/to/this-nodes-data.npz \
  uv run python -m dromeus.node --config examples/node.yaml \
  --factory examples.custom_training:prepare_training
```

The factory is an ordinary importable **local Python function** accepting a
`DraftRunSpec`. You can instead call
`await run_node(config, prepare_training=your_factory)` from your program. Deploy
that program and its dependencies to each participant. Dromeus does not transmit
or execute Python scripts received from peers.

NoLoCo is the default decentralized algorithm. Omitting `algorithm_id` from a
draft is equivalent to `algorithm_id: noloco`; the resolved selection is included
explicitly in the sealed manifest. `algorithm_id: dpsgd` is an explicit
compatibility/control option. NoLoCo settings and artifact codecs must still be
declared consistently for the group; missing settings fail validation.

Your script's optimizer (RMSprop in this example) performs the local inner steps.
Dromeus then computes NoLoCo's outer gradient from the fast and slow weights,
exchanges outer-gradient and slow-weight artifacts with the peer over AXL, and
applies NoLoCo's outer momentum and correction. Choosing a custom model or local
optimizer does not change the group's decentralized algorithm.

Before running, adapt `node.yaml` for each node and replace the environment values
in `custom-regression-draft.yaml` with your deployment's versions. Every node must
use the same draft. Set one node's role to `initiator`; other nodes use
`participant`. The configured AXL bridge must already be running and connected to
the group's peers. Deliver the initiator's invitation file out of band to every
participant at its configured `invitation_path`.

The NPZ contains FP32 `inputs` of shape `[N, 3]` and FP32 `targets` of shape `[N, 1]`.
`N` must be positive and may differ between nodes. Optional `evaluation_inputs`
and `evaluation_targets` contain independently held-out records of those shapes.
The example copies and validates local arrays before contacting AXL. The
application must ensure held-out provenance; Dromeus cannot infer record identity
from arbitrary private data. Changing your model, task, or training policy requires
updating the corresponding definitions and manifest hashes.

The convenient PyTorch interface is:

```python
from dromeus.application import prepare_application
from dromeus.training.trainer import PyTorchTrainer

trainer = PyTorchTrainer(
    model=my_model,
    train_step=my_local_step,       # (model) -> finite loss or None
    evaluate=my_evaluation,         # optional: (model) -> {name: finite value}
    save_state=save_my_state,       # () -> dict[str, numpy.ndarray]
    load_state=restore_my_state,    # (state) -> None
    state_identity=my_local_identity,
)
prepared = prepare_application(
    draft=draft,
    trainer=trainer,
    model_definition=MODEL_DEFINITION,
    task_definition=TASK_DEFINITION,
    training_definition=TRAINING_DEFINITION,
)
```

`train_step` owns one optimizer step and its batch. It can use regression,
classification, or another differentiable objective. Signed losses are valid.
Evaluation is optional and accepts named metrics such as MAE or perplexity; there
is no required accuracy or classification target. New task metrics use evidence
version 2 (`task_round_metrics`); existing M2 evidence version 1 is unchanged.

State callbacks must capture and restore **all** application state required for
the next step: optimizer moments, scheduler, batch cursor, sampler, and any random
generators. Use finite numeric arrays; no pickle is required. `state_identity`
should bind the local records, preprocessing, training settings, device policy,
and other continuation-sensitive choices. Preparation additionally binds the
shared model/task/training definitions, the decentralized algorithm and its
parameters/inner-step count, and the artifact codec configuration. Application
restore binding v2 rejects older, weaker bindings and changed NoLoCo settings.
Same-policy offline continuation under a different run ID remains supported;
changing policy requires a separately audited migration, not ordinary restore.
The adapter saves model state, module
training modes, step count, and last loss. A failed restore rolls back to the old
snapshot; a callback must be able to restore its own saved output. Run failure is
terminal if application code also fails rollback. This is explicit checkpoint
restoration, not automatic distributed restart. NoLoCo also rolls back trainer and
codec state when algorithm-level restoration fails, retaining its outer state and
pending artifacts. A codec or application that cannot restore its own snapshot
causes an explicit fatal rollback error.

The current exchange supports non-scalar FP32 floating model parameters and
buffers. Represent a scalar trainable value as shape `[1]`. Integer buffers stay
local and are checkpointed. Exact tied tensors are supported when their supplied
values agree. Conflicting aliases reject before any model write or state callback.
Ambiguous overlapping views, including overlaps across separate NumPy-backed
storage objects, reject during preparation. Ordinary transposes and disjoint
memory spans remain supported. Model structure and shared parameter meanings must
match on all nodes. Local counts do not change a node's contribution: the current
objective weights nodes equally. Custom models do not imply support for arbitrary
frameworks, heterogeneous parameter schemas, or every optimization algorithm.
Advanced integrations can implement `PreparedTraining` and `CheckpointTrainer`
directly without using the PyTorch convenience adapter. Integrations that bypass
`prepare_application` must bind their own complete configuration and local data
identity in checkpoint state.

The optional classification adapter lives in `dromeus.adapters.classification`.
It provides the earlier validated classification data and optimizer recipe for
callers that choose it. CIFAR loaders, ResNet models, the benchmark registry, and
benchmark configuration live in `benchmarks/workloads/cifar10`; they are excluded
from the installed Dromeus wheel. Repository benchmark commands explicitly use
that workload. Benchmark dependencies are available through the `benchmark` extra
and are included in the repository's development environment.
