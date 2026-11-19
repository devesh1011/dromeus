# Private training

A private run adds record-level differential privacy to every node's local
training. Dromeus owns Poisson sampling, per-record clipping, Gaussian noise and
privacy accounting; the group's optimizer consumes the sanitized gradients.
Install the extra first: `pip install "dromeus[privacy]"`.

## Make a private draft

Pass `privacy` when writing the draft:

```python
dromeus.make_draft(
    task(),
    "draft.yaml",
    run_id="private-regression-2026-11",
    participants=4,
    rounds=20,
    local_steps=10,
    privacy=dromeus.Privacy(max_grad_norm=1.0, noise_multiplier=1.1, delta=1e-5),
)
```

The privacy horizon is the run's total local steps (`rounds x local_steps`), and
sampling switches to Poisson. `epsilon_limit` optionally caps spending. Real
private runs use the default `randomness="secure"`, which needs the pinned secure
RNG dependency; use `"reproducible_public_benchmark"` only for seeded research on
public data.

## Give each node a ledger

Each private dataset needs a durable ledger that records every privacy
reservation, including failed runs. Create it once, then reopen it for every run:

```python
ledger = dromeus.PrivacyLedger.create("ledger.json", lineage="site-a-records")
# later runs: dromeus.PrivacyLedger("ledger.json", lineage="site-a-records")

local = dromeus.LocalData(
    dataset=my_records,
    data_identity=my_file_hash,
    ledger=ledger,
    physical_batch_size=16,  # records processed at once; doesn't change accounting
)
```

Never delete or recreate a ledger to restart a run. Secure runs refuse a
`research_seed`; reproducible public benchmarks require one.

## Register an optimizer

`Optimizer.sgd`, `Optimizer.adam` and `Optimizer.adamw` cover most runs, private
or not. For any other optimizer, pass an `OptimizerRegistration` as the task's
`optimizer`. Every node supplies the same trusted implementation. The
[RMSprop reference](../tests/support/private_optimizer.py) is a complete
registration to copy and adapt. A registration supplies:

- `optimizer_type`: the exact PyTorch optimizer class, including custom subclasses.
- `implementation_hash`: SHA-256 identifying the optimizer implementation, factory
  and state-validator contract. Include relevant dependencies in this identity.
- `create(model)`: construct fresh optimizer state on the actual model parameters.
- `validate_state(state, parameters, logical_steps)`: reject incorrect optimizer
  state fields, tensor shapes and dtypes, counters and initial values.

```python
from my_optimizer import registration

task = dromeus.TrainingTask(
    ...,  # model, batches, loss and definitions as usual
    optimizer=registration(),
    batch_size=64,
    schedule="linear_decay",
)
```

No training or spending happens while the draft is written; the model factory must
construct only public initial state. Group formation shares model weights, never
optimizer moments, sampler state or private data.

The registration's description records the fully qualified class name, implementation hash, Torch
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
