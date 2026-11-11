# Private training with the group's optimizer

Private runs use the [model-independent task interface](model-independent-training.md):
the application describes its model, records and per-record loss as a
`TrainingTask`, and the group's draft adds a `privacy` policy and the task's
`private_training` specification.

Dromeus owns Poisson sampling, per-example clipping, Gaussian noise and accounting.
The group chooses the optimizer that consumes those sanitized gradients. A custom
optimizer does not require editing Dromeus or adding its name to a registry.

The task specification selects either a built-in `PrivateOptimizerSpec` preset
(SGD, Adam or AdamW) or a `FactoryOptimizerSpec` for a registered optimizer. Every
participant supplies the same trusted implementation and matches the complete
shared specification before formation.

## Register an optimizer

[RMSprop reference fixture](../tests/support/private_optimizer.py) is a complete RMSprop registration. Copy and adapt it into your own `my_optimizer.py` module before using the configuration below.
It supplies:

- `optimizer_type`: the exact PyTorch optimizer class, including custom subclasses.
- `implementation_hash`: SHA-256 identifying the optimizer implementation, factory
  and state-validator contract. Include relevant dependencies in this identity.
- `create(model)`: construct fresh optimizer state on the actual model parameters.
- `validate_state(state, parameters, logical_steps)`: reject incorrect optimizer
  state fields, tensor shapes/dtypes, counters and initial values.

Attach the registration to the task and describe it before sealing the group's draft:

```python
from my_optimizer import registration
from dromeus.application import (
    LocalTrainingOptions,
    definition_hash,
    describe_task,
    prepare_task,
)
from dromeus.training.private_trainer import private_training_definition
from dromeus.training.task import TrainingTask

task = TrainingTask(
    model_factory=public_model_factory,
    dataset=open_local_records(),
    batch_adapter=public_batch_adapter(),
    loss_per_record=loss_per_record,
    model_definition=MODEL_DEFINITION,
    task_definition=TASK_DEFINITION,
    loss_definition=LOSS_DEFINITION,
    implementation_hash=INSTALLED_IMPLEMENTATION_SHA256,
    optimizer_registration=registration(),
    public_evaluate=evaluate_public_holdout,
)
spec = describe_task(task, batch_size=64, schedule="linear_decay")
# Set draft.private_training = spec and draft.privacy to the group's policy.
# Set draft.application_training.definition_hash to:
training_hash = definition_hash(private_training_definition(spec))

prepared = prepare_task(
    draft=draft,
    task=task,
    local=LocalTrainingOptions(
        data_identity=local_lineage_identity,
        reservation_store=ledger,
        max_physical_batch_size=16,
    ),
)
```

The surrounding application supplies those data/model/draft/ledger values and the
normal NoLoCo/AXL setup. Use secure randomness for private data, with the pinned
secure RNG dependency installed. For reproducible experiments on public data only,
select that profile explicitly and pass `research_seed`. Missing secure RNG fails
preparation. No training or spending happens during `describe`; its factory must
construct only public initial state. Group formation shares model weights, never
optimizer moments, sampler state or private data.

`describe` returns the fully qualified class name, implementation hash, Torch
version and each group's ordered parameter names and resolved options. Every model
parameter must appear exactly once. Groups may have different learning rates;
linear decay applies the same horizon fraction to each initial rate. Other options
and membership cannot change mid-run. NoLoCo copies weights into existing parameter
objects and retains local optimizer history. Runtime checks reject nested shared
specification changes as well as actual option/group changes.

Dromeus compares the supplied implementation hash to the group's declaration; it
does not discover or audit arbitrary code from that hash. Never import executable
code based on a remote manifest. Install and inspect the optimizer package locally,
then bind the audited factory/validator/package versions to a common digest. The
example hashes its own source and pinned Torch version; custom packages need their
implementation/build identity included too.

## Compatibility contract

The extension supports ordinary dense FP32 `torch.optim.Optimizer` implementations
that update from sanitized `.grad`, model weights, public options and their own
prior state. Their `step()` must work with no closure or extra arguments. The
factory must own the complete model parameter set and resolved options must be
finite JSON values (tuple options such as betas are canonicalized to lists).
Enabled fused/foreach/capturable/differentiable/maximize paths are outside this
profile; when present, declare those flags explicitly false.

Dromeus calls the optimizer once per complete logical batch, including empty
Poisson draws. Raw `grad_sample` and clipped-but-unnoised `summed_grad` attributes
are hidden during the selected optimizer's step and restored for Opacus's internal
lifecycle. Standard Torch zeroing is owned by the delegate; custom `zero_grad`
overrides are not called. Optimizers that depend on custom zeroing, closure-based
recomputation, raw samples/activations, secondary backwards, sparse gradients or
hidden uncheckpointed randomness need a different audited execution contract.

Factories and validators are trusted application code. Attribute shielding reduces
accidental misuse; it is not a sandbox against malicious code or a theorem that
arbitrary optimizer code preserves privacy. Existing private model/loss admission
rules still apply. Pure post-processing of sanitized gradients needs no additional
noise mechanism, but fresh private-data computation must be accounted separately.

## State and failures

Research checkpoints use numeric arrays and bounded metadata, never pickle. Before
restore, Dromeus checks implementation identity, exact group/index mappings, finite
numeric state, scheduled rates and immutable options, then invokes your validator
on detached state/parameter copies in optimizer-group order. A validator must
reject bad shapes/counters and distinguish allowed eager public state at step zero
from prior trained state. Factored/scalar optimizer state is supported when the
validator describes it correctly. `state_dict`/`load_state_dict` must retain all
optimizer history and accept their own output atomically; the callback cannot
repair arbitrary broken optimizer implementations.

Secure runs remain fresh-start only. A failed optimizer step leaves the entire
horizon reservation spent; the trainer refuses further training and model export.
Do not delete/recreate the dataset-lineage ledger to restart a run. Raw private
losses remain suppressed; public evaluation requires declared fixed holdout
provenance. Publish explicit model-only exports, not local training checkpoints.

## Validation scope

Local conformance covers RMSprop, eager-state Adagrad and a custom two-group update
rule against direct Opacus, including sampling/noise identity, microbatch splitting,
empty draws, group schedules, research restore and malformed-state rejection.
Four in-memory NoLoCo nodes exercise the registered optimizer through public
application preparation. CPU secure-mode coverage runs when source-built
`torchcsprng` is available. These tests do not establish GPU/AXL conformance for
every third-party optimizer; validate the implementation your group selects on its
chosen runtime before using it for a private workload.
