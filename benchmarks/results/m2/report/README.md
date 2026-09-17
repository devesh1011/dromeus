# M2 report and review data

Start with [the reviewed report](M2_Gensyn_Submission_Report.pdf), then
[per-run accuracy](metrics.csv), [full metrics and paired deltas](metrics.json),
and the [evidence index](../README.md).

The report presents the NoLoCo implementation, benchmark methodology, figures,
and worker/seed statistics. Appendix B links the release, metrics, training logs,
configuration, software environment, source projects, and artifact availability.
Exact technical identifiers remain in the supporting evidence records.

The [Google Doc](https://docs.google.com/document/d/10zjwEh-Yp9df0_d9XhJQrnXjkPerIIuq9dKUAuXVBCM/edit)
is the editable source. [Artifact availability](../evidence-availability.json)
records which bytes are included, retained separately, or unavailable.

## Regenerate metrics

From the repository root:

```sh
python3 benchmarks/results/m2/report/build_metrics.py
```

This uses only the accepted-run index and retained JSON reports, verifies their
source hashes, and writes `metrics.json` and `metrics.csv`. All 16 accepted runs
are included. Worker dispersion uses population SD; across-seed summaries use
sample SD of run means. Unequal AXL/NCCL seed sets are explicit.

## Regenerate charts

With NumPy and Matplotlib available, run:

```sh
python3 benchmarks/results/m2/report/build_charts.py
```

The builder reads the committed AXL logs and reports without changing them,
validates exactly 500 metrics per worker (42,000 node-rounds), and writes four
figures plus `chart-data.json`. It reads the identity accuracy and nominal byte
baseline from acceptance reports. The JSON retains source hashes, evaluation
rounds and mean accuracies, all 500 mean local-batch losses per run, timing means,
and divergence points, so plot values can also be inspected without parsing logs.

- `learning.png`: worker-mean test accuracy and unsmoothed final local-batch loss.
- `compression.png`: nominal Dromeus transfer-protocol bytes and W4 codec accuracy.
- `divergence.png`: measured weight standard deviation at the frozen cadence.
- `timing.png`: per-node timing means, not synchronized global wall-clock durations.

The 40% pilot timing result in the PDF is historical and does not establish a
controlled timing reduction for the final 45% profile. The report documents
regional differences and the measured scope of the comparison.

## Maintenance boundary

These report builders and the maintained reference harness under
`benchmarks/noloco_reference/` remain included in repository lint and strict type
checks. Only `benchmarks/results/m2/nccl-reference/` is excluded: its Python files
are captured run wrappers and runtime-source snapshots whose exact bytes are
bound by evidence hashes. Do not reformat or repair those archived files in place.
