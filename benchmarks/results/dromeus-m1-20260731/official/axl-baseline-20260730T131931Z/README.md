# AXL baseline

This is the raw evidence behind the AXL transfer results in the M1 report.

1. `public/aggregate/baseline-summary.png` for the chart.
2. `public/aggregate/aggregate.json` for the exact percentiles, goodput, retry counts, and integrity checks.
3. `public/aggregate/rtt-summary.csv` and `transfer-summary.csv` for results broken down by directed node pair.

The `public/raw/` directory contains the measurements before aggregation. Each `node-*` directory has:

- `rtt.json`: 15 RTT samples from that node, covering its three outbound paths with five samples per path.
- `transfer.json`: 75 artifact transfers from that node, covering three destinations, five artifact classes, and five samples per combination.

Across all four nodes, that gives 60 RTT samples and 300 transfer samples over all 12 directed paths. The run recorded three retries and no checksum failures.

## Control files

`public/control/manifest.json` is the sealed manifest for this five-round transport baseline. It records the participants, payload schema, transfer limits, and environment used for the run. This is separate from the three 400-round training manifests elsewhere in the M1 evidence directory.

`public/control/checkpoint.safetensors` is the exact payload used for the `checkpoint` row in the transfer results. It is not one of the final trained model checkpoints.

## Metadata and redactions

`public/aggregate/metadata.json` records the AXL commit and binary hash, Dromeus runtime commit, machine types, AWS regions, and benchmark settings.

The original run is stored at:

`s3://dromeus-m1-benchmark-150911080841/quality/axl-baseline-20260730T131931Z-samples5/`

## Checking the files

From this directory, run:

```bash
shasum -a 256 -c SHA256SUMS
```

Every published file should report `OK`.
