# Dromeus M3 evidence

## Start here

1. [Submission report](https://docs.google.com/document/d/1Yo4Rb8r0LoZ0yBnGHHmjURSV02QLfThFHNdRtumFiDo/edit) — the main narrative, results and charts.
2. [Retained Dromeus logs](audit/README.md) — original event records grouped by experiment, with readable worker filenames.
3. [Milestone requirements](milestone-requirements.md) — what the proposal requires, supporting evidence and remaining gaps.

Per-run written summaries and final-event excerpts are omitted because they
repeat the report or records already present in the logs. Every accepted digit
and warning-assessment repetition remains included. Charts are not duplicated.

## Supporting files

| Location | What it contains |
| --- | --- |
| [results/](results/README.md) | Numeric source tables and retained scientific audit proofs |
| [configurations/](configurations/) | Frozen recipes, calibration selection and the tested warning policy |
| [audit/training-logs/](audit/training-logs/) | 232 editor-readable worker logs; calibration separated from assessment |
| [run-index.json](run-index.json) | Original run IDs, readable labels, log folders and proof references |
| [artifact-availability.json](artifact-availability.json) | Included evidence and unavailable original artifacts |
| [provenance/](provenance/) | Source hashes and exact-copy/derivation labels |
| [verification/](verification/README.md) | Local checks and a portable log verifier |

Folder names are descriptive. Original run IDs, timestamps and event bytes inside
the logs are unchanged. Names such as `private-sigma-0.5-seed-17` identify a
setting and repeat; they do not select the best-performing run.

## Verify

From this directory:

```sh
shasum -a 256 -c checksums.sha256
python3 verification/verify-retained-logs.py
```

The retained logs omit only the declared routine successful transport chatter;
they are not complete, unfiltered originals. Models, checkpoints, raw privacy
histories and credentials are excluded. GPT logs are not included; its retained
results and audits are identified separately.

This follows the [published M2 evidence pattern](https://github.com/devesh1011/dromeus/tree/v0.2.0/benchmarks/results/m2).
Technical evidence is available for the declared profiles. Public tradeoff
publication and Gensyn acceptance remain unverified, and a separate signed
agreement was not located. Report sharing is managed separately. Nothing here
claims independent attestation merely because hashes match.
