# Retained Dromeus training logs

The [submission report](https://docs.google.com/document/d/1Yo4Rb8r0LoZ0yBnGHHmjURSV02QLfThFHNdRtumFiDo/edit)
is the main review document. Use these logs to inspect the original recorded
metrics, warning transitions and completion events behind it.

| Folder | Coverage |
| --- | --- |
| [privacy-utility/](training-logs/privacy-utility/) | 22 accepted digit runs / 88 worker logs |
| [warning-validation/](training-logs/warning-validation/) | 14 held-out warning-assessment runs / 56 worker logs |
| [privacy-calibration/](training-logs/privacy-calibration/) | 10 original calibration runs / 40 worker logs |
| [warning-calibration/](training-logs/warning-calibration/) | 12 warning-calibration runs / 48 worker logs |

Each run folder contains `worker-0.jsonl` through `worker-3.jsonl`. Open them
directly in a text editor. For example:

```text
training-logs/privacy-utility/private-sigma-0.5-seed-17/worker-0.jsonl
training-logs/warning-validation/no-fault-non-private-seed-113/worker-0.jsonl
```

The [log index](log-index.json) maps the friendly folders to original run IDs,
worker identities, source hashes and event counts. Original identifiers inside
the records have not been renamed. Retry suffixes and distinct calibration
settings remain explicit. A secure run's model seed is not its private noise seed.

## What was retained

These 232 files contain 190,041 retained original event records. The historical
retention step omitted only `message_received` rows with `routed` exactly true
and `transfer_message_sent` rows with integer `retry_count` exactly zero. Rows
with error/failure fields were retained. Metrics, evaluations, consensus
observations, warnings, retries and completion events were preserved.

The files were decompressed and renamed without rewriting their records. They
are not reconstructed logs, but they cannot restore omitted successful
per-message traffic. Calibration runs are not held-out repetitions, and completion
of training does not mean the original warning policy passed.

## Additional supporting evidence

- [Original warning-policy failures](../results/warning-detection/earlier-policy-failures.json)
- [Historical failed attempts](../results/private-training/historical-failed-attempts.json)
- [Fresh warning assessment](../results/warning-detection/assessment.json)
- [Validated warning policy](../configurations/divergence-policy.json)
- [Calibration selection](../configurations/calibration-selection.json)
- [Artifact availability](../artifact-availability.json)

Complete failed-attempt bundles and GPT logs are not supplied. Their retained
summaries/audit proofs remain labelled as such; later successes do not erase
adverse results.

## Verify

From the M3 evidence root:

```sh
python3 verification/verify-retained-logs.py
shasum -a 256 -c checksums.sha256
```

The log verifier checks hashes, identities, 75 rounds, five evaluations and
completion for every worker, then reconciles 2,552 numeric values with the digit
CSV inputs. It respects the original calibration consensus-event schema.
These checks do not train models or query AWS. Hashes establish consistency with
retained evidence, not independent proof of execution.
