# Divergence monitoring

Any run can opt into `divergence_detection` with a `DivergencePolicy` in its draft.
Every threshold is explicit; the library does not install a universal default.
The M3 [October 25 repair report](https://github.com/devesh1011/dromeus/blob/v0.3.0/benchmarks/results/m3/divergence-repair-2026-10-25/REPORT.md)
validates the growth-ratio-3 policy for its declared small digit workload only.
It recorded zero false warning entries across 11 no-injection groups and detected
all three injected-fault groups within two committed rounds. Each group's four
monitors are correlated. Any other thresholds remain exploratory until calibrated.

## Use the validated digit policy

[`divergence-policy.json`](../benchmarks/results/m3/configurations/divergence-policy.json) is an exact
copy of the measured policy. Its canonical policy hash is
`d7a815d67ff04c9419231b89a400d63f2e951063c1c37ff67fef699cd407a448`, and its
threshold-set ID remains `m3-repair-candidate-growth-ratio3`. Keeping that measured
ID preserves the link to the live evidence; the word `candidate` does not change
its recorded validation status.

Use it only for the recorded four-worker public CNN digit recipe: MNIST, USPS,
SVHN and Optical Digits; 75 NoLoCo rounds with 10 local steps per round; ordinary,
private sigma 1/sigma 4, and the declared growing-offset fault. Compression and
unequal-data/label-skew each received one auxiliary smoke check. Consult the
report and the frozen recipes at tag [`v0.3.0`](https://github.com/devesh1011/dromeus/blob/v0.3.0/benchmarks/results/m3/README.md)
before reusing the profile. It does not establish
thresholds for text models, other recipes, secure private datasets, or all failure
families. Low disagreement can still accompany poor accuracy or shared collapse.

Add the policy when writing a **new** draft for that workload. From the repository
root:

```python
import hashlib
from pathlib import Path

import dromeus

source = Path("benchmarks/results/m3/configurations/divergence-policy.json")
if hashlib.sha256(source.read_bytes()).hexdigest() != (
    "f8d8fda6688c750e1cd679f6ed81e8f12e9607080162e47c3b4222d2b4955aa5"
):
    raise ValueError("digit policy file differs from the validated policy")
policy = dromeus.DivergencePolicy.model_validate_json(source.read_bytes())

dromeus.make_draft(
    digit_task(),  # your task for the recorded digit recipe
    "digits-with-warnings.yaml",
    run_id="digits-with-warnings-2026-11",
    participants=4,
    rounds=75,
    local_steps=10,
    axl_version="your-pinned-axl-version",
    divergence=policy,
)
```

Give every participant the same draft, and set each node configuration's
`draft_path` to `digits-with-warnings.yaml`. The policy belongs in the shared draft,
not at the top level of `node.yaml`. Formation binds it into the sealed draft hash;
the runtime reads that sealed policy when constructing its divergence monitor.
Choose a fresh run ID and local run roots before formation. Preserve archived
manifests, policies and checkpoints; changing a threshold changes shared run
identity and is not an in-place policy update for an existing run.

The example JSON and benchmark evidence are repository artifacts and are not
included in the runtime-only wheel. Supply the reviewed JSON with your deployment.
Schema support does not validate these numeric thresholds for any other task.

## How warnings work

The detector consumes complete, committed-round consensus observations and tracks
`warming_up`, `healthy`, `suspect`, `diverging`, or `insufficient_data`. These are
local diagnostic states, not proofs of convergence or useful learning. Warnings do
not stop training, coordinate peers, or change hyperparameters.

For each full consecutive window after warm-up, it compares the median of the
first half with the median of the second half and measures the slope of the log
distance against round IDs. All three growth conditions must hold: later median
above the floor, median ratio above `growth_ratio`, and slope above
`min_log_slope`. `patience_windows` consecutive qualifying endpoints activate a
warning. A separate `severe_distance`/`severe_patience` rule catches large persistent
distances. Recovery needs `recovery_windows` full windows entirely below
`recovery_distance`, with no growth condition. Recovery thresholds must be below
severe thresholds.

The local committed round acts as a watermark. Observations arriving out of order
are buffered within `max_lag_rounds`; a missing round strictly older than that
allowance becomes a permanent gap. A gap resets the consecutive window and all
patience/recovery streaks. An existing warning remains latched until valid recovery
evidence arrives. Duplicate or finalized late observations cannot advance patience
or rewrite warnings. A jump in committed progress does not require iterating over
all missing rounds. Windows and pending buffers have explicit policy bounds.

Every run records one `consensus_observation` evidence record per complete round,
with normalized RMS distance, absolute RMS spread, mean-sketch norm and a
denominator-degenerate flag (mean norm <= 1e-12). A near-zero mean triggers a separate scale warning. Optional
`absolute_distance_floor`, `absolute_severe_distance` and
`absolute_recovery_distance` must be supplied together to assess divergence using
absolute spread in these windows. Without them, such windows are insufficient data
for a divergence/recovery decision.

Each node's evidence log (`run_root/logs/dromeus.jsonl`) records every
state, warning and scale transition once as a `divergence_status` record, with the
policy hash, threshold-set identity, sequence number, contributing window and
measurements. Gaps in sequence numbers indicate omitted records.

Metrics, consensus and warnings share one bounded background event queue, so a
slow or failed log write cannot block training. Local detector state updates even
if the queue overflows, and shutdown is bounded.

The M2 replay study and its exploratory thresholds are recorded with the M3
evidence at tag [`v0.3.0`](https://github.com/devesh1011/dromeus/blob/v0.3.0/benchmarks/results/m3/README.md); its replay
tooling is not maintained on `main`.

## Plot each node's actual distance

Every run saves `node_consensus_distance` evidence automatically, even with no
`divergence_detection` policy. No additional user setting is required.
Each record measures the logging node's committed model sketch against the mean
of **all sealed participants' sketches for that round**. It includes:

- `node_id`: the measured node's sealed public key.
- `absolute_distance`: L2 distance from that node's sketch to the group mean.
- `normalized_distance`: absolute distance divided by `(mean_sketch_norm + 1e-12)`.
- `group_normalized_rms`, `mean_sketch_norm`, `sketch_count` and
  `denominator_degenerate`: context for interpreting the measurement.

These are approximate model distances in CountSketch space. They do not measure
accuracy, establish convergence, or supply calibrated warning thresholds. A small
mean norm can make normalized distances large; inspect the absolute-distance
panel and red scale-warning bands alongside the normalized panel.

The runtime computes a measurement only after every sealed participant's sketch
arrives. Missing inputs produce missing rounds, not partial-group measurements.
The member mapping uses public keys, regardless of arrival order. Measurements
reuse already exchanged sketches and do not add network messages or privacy
mechanisms.

All evidence shares the bounded background queue, including runs without
detection. Missing evidence can therefore reflect missing sketches or delivery
loss; it does not mean zero divergence.

The per-node plotting tool used for the M3 study, and the source-bound logs it
plotted, are available at tag [`v0.3.0`](https://github.com/devesh1011/dromeus/blob/v0.3.0/benchmarks/results/m3/divergence-repair-2026-10-25/REPORT.md).
