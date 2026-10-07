# M3 matched non-private baseline

The canonical M3 guide in Obsidian owns experiment decisions. These tools prepare
and run a baseline; they do not freeze a noise grid or claim final M3 results.

`matched_baseline.MatchedBaseline` is a benchmark adapter around the normal
`PyTorchTrainer`. It accepts the group's model and optimizer factories, fixed
public tensors and a `BaselineRecipe`. It uses Opacus 1.6.0's Poisson loader
without wrapping the optimizer in a privacy engine. There is no per-example
clipping, added noise, accountant or privacy reservation. Reports label it
`non_private_unclipped` and set epsilon to `null`.

The recipe binds batch request, loss, horizon, schedule, evaluation cadence and the group's optimizer
semantic definition. The optimizer definition must identify the implementation,
version, all options, parameter groups and learning rates. The application owns
that declaration and factory; Dromeus does not infer custom optimizer semantics.
The generic baseline accepts compatible PyTorch optimizer factories, including
RMSprop and custom implementations. They must own all model parameters exactly
once, support an ordinary `step()` using accumulated gradients, and expose numeric
checkpoint state. Models and evaluators must be deterministic per example; any
optimizer randomness must be owned and included in its numeric checkpoint state.
Built-in Dropout modules reject before preparation. Hidden randomness in trusted
custom forward/optimizer/evaluator code is outside this benchmark contract.
Closure-based raw-data recomputation is outside this benchmark
step interface. Extending the production private adapter's existing optimizer
allowlist remains a separate implementation task.

## What is matched

Use the same model and public initialization, local data/preprocessing, loss,
optimizer/options, logical steps, learning-rate schedule, outer NoLoCo settings,
codecs, public holdouts and evaluation cadence as the DP comparison arm.

Sampling follows the owned private adapter: start a new Poisson-loader iterator
for each logical step and consume its first independent draw. Resolve
`q = 1 / ceil(N / requested_batch_size)` from the returned loader, then divide
summed per-example loss gradients by `int(N * q)`, matching the DP optimizer.
For example, N=3,000 and requested batch64 give q=1/47 and expected batch63.
Actual batches vary; do not divide by their observed size or redraw empty batches.
An empty draw performs a zero-gradient optimizer step and advances the schedule,
so momentum/weight decay still apply. Physical chunks accumulate one logical
gradient; only one optimizer update occurs. MSE averages features per example;
cross-entropy sums examples before the expected-batch normalization.

The baseline and reproducible public DP arm can use the same local sampling seed.
A secure DP run has a different random stream; match its sampling rule/rate and
work budget, not its exact draws. Public benchmark model factories must have
per-example behavior compatible with the DP comparison (the digit model uses
GroupNorm). Baseline compatibility does not certify a custom model/optimizer for
private use.

Numeric research checkpoints retain optimizer, sampler RNG and logical/physical
counters. Outer weight loads preserve optimizer state. The baseline's shared
training definition and local checkpoint identity bind the recipe and local data.
`matched_draft(reference, recipe, run_id=...)` preserves model/task/environment,
NoLoCo/codec/transport settings, rounds and local steps; it removes privacy fields,
requires a distinct run ID and validates batch/loss/schedule/optimizer agreement
when the reference has a private training specification.

## Public digit runner

`matched_digits.py` uses the CNN, preprocessing and four public sources from the
previous digit pilot. It consumes an explicit private **reference draft** for the
next comparison. The previous shuffle-based pilot draft is not a DP reference.
The reference must bind this model/task, the selected initialization seed and
public fixed holdout settings. The digit runner adapts the currently implemented
private optimizer specification; its built-in factory covers SGD/Adam/AdamW. Use
the generic baseline factory interface for other compatible optimizers; the
private adapter must also support them before a DP comparison can be claimed. Choose the reference horizon/settings after the
budget/resource forecasts; these tools do not choose them for you.

Prepared data files contain FP32 `inputs` in [0,1], shape `[N,1,32,32]`, and int64
`labels` 0..9. Rank 0/1/2/3 owns MNIST/USPS/SVHN/Optical Digits respectively. Each
node opens only its own `<source>-train.npz` and all four public `<source>-test.npz`
files. Inputs are scaled to [-1,1]. Data remains external to the runner's output.

Install Opacus 1.6.0 through the optional privacy profile. For a local comparator:

```bash
uv run --no-sync python -m benchmarks.m3.matched_digits local \
  --reference /path/to/private-reference-draft.json \
  --data /path/to/prepared-digits --root /path/to/new-baseline-node-0 \
  --run-id digits-poisson-baseline-s17 --rank 0 --seed 17 \
  --physical-batch 32 --device cpu --evaluation-interval 25
```

Repeat with ranks 1–3. Initialization uses the shared seed; the public sampler
uses seed + rank. `--evaluation-interval` counts outer rounds (local evaluation
interval is that value times local steps). Match it to the DP arm.

For collaborative NoLoCo, place a normal Dromeus `node.json` in each fresh output
directory and use `node` instead of `local`. Its `run_root` must be that directory
and its `draft_path` must be `<root>/draft.json`, which this runner materializes.
Use the normal initiator/participant roles, shared invitation path/bootstrap URI
and each machine's AXL bridge URL. All nodes must derive identical baseline drafts
from the same reference/run ID/recipe. The runner passes the prepared baseline
through `run_node`, verifies terminal completion and the logical-step horizon,
and writes final public model-only weights. This command connects to an existing
AXL setup; it does not provision machines.

Each output retains `draft.json`, `initial.safetensors`, `baseline-model.safetensors`
and `baseline.json`. The summary includes reference/baseline hashes, checkpoint
and dataset/holdout hashes, actual sampler rate/expected batch size, logical and
physical counts, empty draws, public seeds, recipe and available evaluation
history, training/evaluation wall time and CUDA peak allocation when applicable.
Collaborative intermediate evaluations and per-node distances are in
the normal runtime logs/run-store. Run-store checkpoints are local training state;
use the explicit model file for a published model artifact.

Use fresh output directories. A node failure leaves partial runtime evidence and
no completed `baseline.json`; do not publish it as a complete comparison. Reports
must reconcile all four nodes and compare the DP arm's initial/data/configuration
hashes and actual sampler rates before plotting. Full real-AXL/GPU baseline
execution and utility curves remain W3/W4 validation work.

## Verification

```bash
uv run --no-sync pytest tests/benchmarks/test_matched_baseline.py -q
```

These tests check an analytic unclipped update, pinned Opacus zero-noise/large-bound
reference trajectories for MSE/cross-entropy, resolved sampler rounding, empty
draws, microbatch equivalence, RMSprop state/restore, counter rejection, digit
runner provenance, reference mismatch rejection and four in-memory NoLoCo nodes.
The zero-noise reference is a non-private diagnostic, not a DP setting.
