# Dromeus

Dromeus is a Python library for **decentralized federated learning**. Each
participant trains on its own local dataset and exchanges model updates with
peers over [Gensyn's AXL network](https://github.com/gensyn-ai/axl), without a
central server aggregating those updates.

You bring the model, per-record loss and optimizer as a `TrainingTask`. Dromeus
coordinates the group, runs local training and applies
[NoLoCo](https://github.com/gensyn-ai/noloco) updates between peers. It is built for developers experimenting with decentralized training on
independently operated machines.

## How training works

Participants agree on a model, training policy, and fixed membership before the
run starts. They then repeat the same loop:

1. Train on local data for the configured number of steps.
2. Exchange outer gradients and slow weights with the scheduled peer.
3. Apply NoLoCo's outer momentum update and save the committed round.

The peer schedule changes each round. Pairs make progress independently, subject
to readiness and timeout limits. One node coordinates group setup; training
updates happen between peers without a global all-reduce.

AXL runs beside each application and handles encrypted network routing. Dromeus
adds chunk acknowledgments, bounded retries, and checks on the sender, round,
and tensor contents. Top-k sparsification and 8-bit quantization reduce update
traffic, while error feedback carries outer-gradient compression error into
later rounds. An identity codec is also available.

Each worker keeps its own model, and final weights can differ. Per-node logs
record evaluation metrics, timing, and transfer behavior.

## Get started

Use Python 3.12 and `uv`. Install from the source checkout:

```bash
git clone https://github.com/devesh1011/dromeus.git
cd dromeus
./scripts/bootstrap
```

The [training guide](guides/application-training.md) covers the three steps
with a complete regression example:

```python
import dromeus

task = dromeus.TrainingTask(...)  # model, batches, loss, optimizer; same on every node
dromeus.make_draft(task, "draft.yaml", run_id="my-run", participants=4,
                   rounds=20, local_steps=10, axl_version="your-axl-version")
# on each node, with its own records:
await dromeus.Node.from_file("node.yaml").run(task, local=dromeus.LocalData(...))
```

Before launching, configure an AXL node on each machine and connect the group's
peers. The [configuration guide](guides/configuration.md) covers each node's
`node.yaml`. One participant initiates group formation; share its invitation file
with the other participants. From the command line, each node runs:

```bash
dromeus doctor --config /path/to/node.yaml   # optional: is this machine ready?
dromeus run --config /path/to/node.yaml --task my_training:setup
```

where `my_training.setup()` returns this node's `(task, LocalData)`. The
[command-line guide](guides/command-line.md) covers every command and exit code.

## Working assumptions

Dromeus supports fixed, even-sized groups of 4 to 16 participants. Nodes must
agree on model structure and parameter meaning; the PyTorch integration exchanges
compatible FP32 model state. Participants may hold different amounts of data,
but each node has equal weight in the outer update.

Membership stays fixed for the run. A peer failure or missed deadline can end
training; automatic distributed restart is not implemented. Checkpoint restore
requires compatible application state.

Keeping datasets local does not make ordinary model updates private. Optional
private training uses Opacus per-example clipping, Gaussian noise and per-node
accounting through an owned trainer. Groups supply compatible optimizer factories;
private execution also requires an admitted model/loss and the documented state
contract. See [private training](guides/private-training.md) for supported
profiles, secure randomness, privacy scope and fresh-start restrictions.

Divergence monitoring records model disagreement and emits configurable warnings.
Thresholds are workload-specific and require calibration; agreement alone does not
prove model utility. See [monitoring](guides/divergence-monitoring.md).
Secure aggregation and Byzantine fault tolerance are not implemented.

## Benchmarks

The [technical report](https://github.com/devesh1011/dromeus/releases/download/v0.2.0/M2_Gensyn_Submission_Report.pdf)
compares NoLoCo over AXL with an independent NCCL reference at 4, 8, and 16 GPU
workers using GroupNorm ResNet-18 on IID CIFAR-10 partitions. It includes codec
comparisons, learning curves, communication measurements, and deployment limits.
These results describe that workload; they do not establish convergence for
arbitrary models or data distributions.

See the [accuracy table](https://github.com/devesh1011/dromeus/blob/v0.2.0/benchmarks/results/m2/report/metrics.csv)
and [published M2 evidence index](https://github.com/devesh1011/dromeus/blob/v0.2.0/benchmarks/results/m2/README.md)
for retained metrics, charts and frozen settings. The owner permanently retired
the old local raw-result trees; unique deleted logs and models are no longer
available for re-audit from this checkout.

The [M3 team review](https://github.com/devesh1011/dromeus/blob/v0.3.0/benchmarks/results/m3/README.md)
maps the proposal requirements to numeric results, runtime and privacy checks, and
the validated digit-warning policy. It also includes the completed language-model
study as additional evidence. The folder omits report charts and large training
artifacts. Read the [criteria check](https://github.com/devesh1011/dromeus/blob/v0.3.0/benchmarks/results/m3/milestone-requirements.md)
for verified scope, publication status, and evidence-retention limits.

Each milestone's benchmark code and evidence are verified at its release tag
(`v0.1.0`, `v0.2.0`, `v0.3.0`). `main` carries only the current library and its
tests; the benchmark runners used for those milestones are not maintained here.

## Development

Run `./scripts/verify` before contributing. It checks architecture boundaries,
lint, types, and tests. The [test guide](tests/README.md) lists focused commands
and the prerequisites for real-AXL integration tests.

## License

[MIT](LICENSE).
