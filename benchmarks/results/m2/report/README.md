# M2 report and review data

Start with [the reviewed report](M2_Gensyn_Submission_Report.pdf), then
[per-run accuracy](metrics.csv), [full metrics and paired deltas](metrics.json),
and the [evidence index](../README.md).

The PDF is the 14-page revision saved from Google Docs on 2026-09-16. It retains
all four figures, adds worker/seed statistics and reviewer links, and preserves
the existing NCCL-origin and repetition wording at the user's instruction.
The [working Google Doc](https://docs.google.com/document/d/10zjwEh-Yp9df0_d9XhJQrnXjkPerIIuq9dKUAuXVBCM/edit)
remains the editable document. Appendix B's public-access statements are a
snapshot from before the current evidence commits; [availability](../evidence-availability.json)
and the repository index describe the files now packaged for review.

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
controlled timing reduction for the final 45% profile. No normalized cross-fabric
speed, new release publication, or final grant acceptance is claimed.
