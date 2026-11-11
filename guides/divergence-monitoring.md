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

Before formation, add the policy to a **new, otherwise complete draft** for that
workload. From the repository root:

```python
from pathlib import Path

import yaml

from dromeus.manifests.canonical import canonical_hash, parse_draft_yaml
from dromeus.manifests.models import DivergencePolicy, DraftRunSpec

policy = DivergencePolicy.model_validate_json(
    Path("benchmarks/results/m3/configurations/divergence-policy.json").read_bytes()
)
if canonical_hash(policy) != (
    "d7a815d67ff04c9419231b89a400d63f2e951063c1c37ff67fef699cd407a448"
):
    raise ValueError("digit policy differs from the validated policy")

draft = parse_draft_yaml(Path("new-digit-draft.yaml"))
values = draft.model_dump(mode="python")
values["divergence_detection"] = policy
draft = DraftRunSpec.model_validate(values)
Path("digits-with-warnings.yaml").write_text(
    yaml.safe_dump(draft.model_dump(mode="json", exclude_none=True)),
    encoding="utf-8",
)
```

Give every participant the same generated draft, and set its node configuration's
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

`runtime.divergence_status` exposes the immutable current snapshot, including the
active warning, source/detection rounds, window, age and coverage.
`runtime.divergence_dropped` exposes the shared detector-enabled event delivery
drop count. `divergence_status` evidence records each state/warning/scale transition
once with policy hash, threshold-set identity, sequence, contributing window and
measurements. Gaps in delivery sequence numbers indicate omitted records.

When the node has an event sink, node metrics, consensus and warnings share one bounded
background event sink. A slow, failed or locked underlying sink cannot block the
training loop. Local detector state updates even if the event queue overflows.
Shutdown is bounded; an already blocked sink write may complete later, but no new
writes start after its timeout. Callers adding their own event producers should
use `runtime.event_sink` rather than writing directly to its underlying sink.
Call `runtime.stop()` to release event delivery, including when formation fails.

### Require final sketches before completing a run

Large models can take longer to project than the default five-second telemetry
shutdown window. Applications that require the final consensus observation can
opt into a completion policy on `TrainingConfig`:

```python
from dataclasses import replace

from dromeus.runtime import ConsensusCompletionPolicy

training = replace(
    training,
    consensus_completion=ConsensusCompletionPolicy(
        timeout_seconds=120.0,  # choose and validate for your deployment
        barrier=wait_for_group_completion,
    ),
)
```

`wait_for_group_completion(receipt, deadline)` is an application-supplied async
function. The runtime first waits for final sketch publication, receipt of every
sealed member's final sketch, and completion of its final consensus callback.
It then passes a receipt binding the run, manifest, local node, final round and
participant keys to the barrier. Receivers remain active while the barrier runs.
The deadline is an absolute time on the current asyncio event loop's clock; both
stages share that deadline.

The barrier must verify a matching local-completion receipt from **every** sealed
participant before returning. A successful local send, an empty queue, or this
node's complete observation does not establish that the other nodes received its
sketch. Use trusted, run-scoped coordination and reject stale, conflicting or
incomplete receipts. The library does not supply that coordination service or
authenticate arbitrary application receipts on your behalf.

Timeout, missing peers, callback failure or cancellation prevents successful run
completion. Already committed updates and privacy reservations remain consumed;
this is not permission to replay training or resume a secure private run. This
policy covers final consensus computation and reporting, not delivery of every
queued event to an external sink. With no policy, existing best-effort shutdown
behavior is unchanged.

Opted-in cleanup uses the same absolute deadline and reports incomplete shutdown
instead of waiting indefinitely for a blocked report callback. Callbacks should
support cancellation and bound their own I/O. Canceling an asyncio task cannot
kill a native projection thread or arbitrary callback code. Telemetry's
`pending_shutdown_tasks` counts unfinished asyncio tasks only;
`projection_thread_may_continue` conservatively records interrupted threaded
work. Deployments still need a process-level deadline for uncooperative code.

Run the focused checks:

```bash
uv run --no-sync pytest tests/unit/telemetry tests/unit/runtime/test_divergence_runtime.py -q
```

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

All event producers share the bounded background sink, including
runs without detection. `runtime.telemetry_delivery_dropped` reports lost/failed
event deliveries independently of the detector. Missing evidence can therefore
reflect missing sketches or delivery loss; it does not mean zero divergence.

The per-node plotting tool used for the M3 study, and the source-bound logs it
plotted, are available at tag [`v0.3.0`](https://github.com/devesh1011/dromeus/blob/v0.3.0/benchmarks/results/m3/divergence-repair-2026-10-25/REPORT.md).
