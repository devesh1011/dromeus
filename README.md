# Dromeus

Dromeus is a Python library for training models across machines that each hold
their own data. Each participant trains locally and exchanges model updates with
a peer over [Gensyn's AXL network](https://github.com/gensyn-ai/axl).

You bring the model, data loader, loss, and local optimizer. Dromeus coordinates
the group and applies [NoLoCo](https://github.com/gensyn-ai/noloco) updates between
peers. It is built for developers experimenting with decentralized training on
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

The [application guide](examples/README.md) walks through a custom PyTorch
regression model using Huber loss and RMSprop. Each participant supplies its own
local data file. You can replace that training code with your own workload.

Before launching, configure an AXL node on each machine and connect the group's
peers. Adapt [the node configuration](examples/node.yaml) for each participant,
fill in the shared settings in [the run draft](examples/custom-regression-draft.yaml),
and prepare the local data described in the guide. One participant initiates
group formation; share its invitation file with the other participants.

Once those inputs are ready, launch each participant:

```bash
DROMEUS_LOCAL_DATA=/absolute/path/to/this-nodes-data.npz \
  uv run python -m dromeus.node \
  --config examples/node.yaml \
  --factory examples.custom_training:prepare_training
```

The factory connects your local training code to the shared run. The guide
covers optional evaluation and the state callbacks needed to restore your
optimizer and data loader.

## Working assumptions

Dromeus supports fixed, even-sized groups of 4 to 16 participants. Nodes must
agree on model structure and parameter meaning; the PyTorch integration exchanges
compatible FP32 model state. Participants may hold different amounts of data,
but each node has equal weight in the outer update.

Membership stays fixed for the run. A peer failure or missed deadline can end
training; automatic distributed restart is not implemented. Checkpoint restore
requires compatible application state.

Keeping datasets local does not make model updates private. Dromeus does not
currently provide differential privacy, secure aggregation, or Byzantine fault
tolerance. The application guide describes the supported tensor layouts and
other integration constraints.

## Benchmarks

The [technical report](benchmarks/results/m2/report/M2_Gensyn_Submission_Report.pdf)
compares NoLoCo over AXL with an independent NCCL reference at 4, 8, and 16 GPU
workers using GroupNorm ResNet-18 on IID CIFAR-10 partitions. It includes codec
comparisons, learning curves, communication measurements, and deployment limits.
These results describe that workload; they do not establish convergence for
arbitrary models or data distributions.

See the [accuracy table](benchmarks/results/m2/report/metrics.csv) and
[evidence index](benchmarks/results/m2/README.md) for per-run results, logs,
configuration, and artifact availability.

## Development

Run `./scripts/verify` before contributing. It checks architecture boundaries,
lint, types, and tests. The [test guide](tests/README.md) lists focused commands
and the prerequisites for real-AXL integration tests.

## License

[MIT](LICENSE).
