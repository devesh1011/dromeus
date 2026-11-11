# Application scripts with ordinary or private training

Every participant installs the same trusted Python module and supplies its own
local records. The filename is arbitrary. Dromeus prepares the task, coordinates
NoLoCo, and exchanges parameter-update bytes through AXL.

```sh
python -m dromeus.node --config node.json --factory my_training:prepare
```

Task runs use manifest version 7, like every run. The callback interface in
[application-owned training](application-training.md) remains available when your
script must own its own optimizer step.

## What the application supplies

`TrainingTask` accepts a model factory, finite indexed dataset,
`TensorBatchAdapter`, per-record loss function, definitions, and an optional
trusted optimizer registration. Custom model class names require no core edits.
The first batch adapter supports dictionaries, tuples and lists of dense tensors
with record axis zero, including integer token IDs and boolean masks. Shapes after
the record axis are fixed; pad sequences to a declared public length. Public
non-batched constants can live in the model/task definition rather than the batch.

```python
from dromeus.application import LocalTrainingOptions, prepare_task
from dromeus.training.task import TrainingTask


def prepare(draft):
    task = TrainingTask(
        model_factory=build_model,
        dataset=open_local_records(),
        batch_adapter=public_batch_adapter(),
        loss_per_record=loss_per_record,
        model_definition=MODEL_DEFINITION,
        task_definition=TASK_DEFINITION,
        loss_definition=LOSS_DEFINITION,
        implementation_hash=INSTALLED_IMPLEMENTATION_SHA256,
        optimizer_registration=optimizer_registration(),
        record_unit="fixed_token_sequence",
    )
    return prepare_task(
        draft=draft,
        task=task,
        # For an ordinary draft, pass the shared TaskTrainingSpec explicitly.
        specification=shared_specification(),
        local=LocalTrainingOptions(
            data_identity=local_lineage_and_version(),
            reservation_store=local_ledger() if draft.privacy else None,
            max_physical_batch_size=physical_limit(),
            research_seed=None,  # Secure private mode; ordinary defaults to seed 0.
        ),
    )
```

This is an interface sketch. The script supplies the forward/loss computation,
while Dromeus invokes it and owns backward and optimizer updates.
Return a differentiable tensor `[B]`, one scalar per privacy record. Do not call
backward or step an optimizer inside the callback. Do not mix records, read other
private examples, retain raw activations across steps, or publish raw losses.

Use `describe_task(task, batch_size=..., optimizer=...)` to construct the public
`TaskTrainingSpec` before sealing. With a task optimizer registration, omit the
`optimizer` argument; otherwise pass a built-in `PrivateOptimizerSpec` preset. `describe_task` examines a public model and declared
batch schema without reading dataset records. `prepare_task` validates definitions
against the group's manifest, then selects private or ordinary execution.

Private mode requires positive clipping/noise and an explicit privacy policy,
finite population, persistent ledger and Poisson sampling. Ordinary mode can use
`sampling="shuffle"` without Opacus. A matched baseline deliberately selects
Poisson sampling and therefore uses the optional privacy dependency's sampler,
without clipping, noise, accounting or a privacy claim.

## Token records and privacy ledgers

Language-model tasks fit the same interface: store fixed-length integer token
records, pad to a declared public length, and freeze tokenizer vocabulary and
sequence boundaries before training. A fixed-sequence privacy statement does not
automatically protect an entire document or person; repeated or overlapping chunks
need composition.

Private runs need a persistent ledger per dataset lineage. The operator creates it
explicitly with `PrivacyLedger.create(path, lineage=...)`; existing ledgers must
carry all applicable reservations, including failed runs. For public-data
reproducibility only, select the `reproducible_public_benchmark` randomness profile
and pass `research_seed`; secure mode rejects a seed and requires CSPRNG.

## Supported execution and validation

The current generic private profile uses pinned Opacus hooks, dense FP32
parameters, flat clipping, fixed record semantics and deterministic task/model
execution. It supports custom containers composed of compatible operations,
shared parameters, declared frozen parameters and immutable public container
buffers. It checks a public/synthetic fixture against independent per-record
autograd before private records are sampled. This is evidence for the declared
profile, not a proof that arbitrary Python has no data-dependent side effects.

Unsupported operations name their module path and missing capability. BatchNorm,
cross-record losses, unsafe trainable-layer buffers, uncontrolled model randomness,
unbounded streams and changing record populations require different profiles.
Opacus custom gradient samplers are trusted local code and must satisfy the same
conformance checks. There is no silent layer replacement or non-private fallback.

Each tied parameter is clipped and exchanged once. Public immutable buffers and
frozen weights are excluded from NoLoCo/codec updates and verified locally.
Consequently, consensus for a partially frozen task measures trainable coordinates.
Model-only exports retain permitted model state, including aliases and declared
public constants, so they can be loaded into the same application architecture.
Local research checkpoints also contain optimizer/sampler/accountant state and must
not be published. Secure crash resume remains unsupported.

An empty Poisson draw still performs one noise-only logical update. Physical
microbatches do not add accounting steps. Failed updates retain the complete
reservation, and raw private loss is suppressed before telemetry sees it.

## Trust and identity

Implementation hashes and shared definitions establish agreement among peers;
they do not remotely attest code or sandbox a malicious local process. Install
and inspect the application and dependencies locally. Dromeus never imports code
named by a peer's manifest. Public initialization, public evaluation and the
absence of extra private-data releases remain part of the privacy contract.
