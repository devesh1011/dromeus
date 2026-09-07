# CIFAR benchmark workload

This package owns the CIFAR data source, preprocessing and partitioning, ResNet-32
and GroupNorm ResNet-18 models, and the benchmark model registry. These files are
excluded from the installed Dromeus wheel.

`runtime.py` is the benchmark's explicit workload factory and node wrapper. It
uses the same `dromeus.node.run_node` interface as a developer's application.

- `benchmarks/noloco`: M2 launch, experiment selection, instrumentation and reports.
- `benchmarks/noloco_reference`: independent optimizer/control implementation.
- `src/dromeus`: production training interfaces, algorithms, gossip and AXL runtime.

Moving the workload does not move training out of Dromeus. Official Dromeus runs
still execute `src/dromeus/algorithms`, `gossip`, `transport`, and `runtime`.
The frozen model definitions, experiment values and retained run artifacts are
preserved. See `examples/README.md` for a developer-owned custom model and dataset.
