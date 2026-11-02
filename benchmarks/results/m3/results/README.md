# Supporting measurements and audit proofs

The [submission report](https://docs.google.com/document/d/1Yo4Rb8r0LoZ0yBnGHHmjURSV02QLfThFHNdRtumFiDo/edit)
contains the presentation. These files support its numbers and verification;
there are no duplicate charts or per-run narrative summaries.

| Folder | Main evidence |
| --- | --- |
| [privacy-utility/](privacy-utility/) | [Accuracy versus privacy](privacy-utility/accuracy-vs-privacy.csv), [worker results](privacy-utility/worker-results.csv), [learning trajectories](privacy-utility/learning-trajectory.csv), [study aggregate](privacy-utility/study-results.json) |
| [private-training/](private-training/) | Supported execution profiles, secure-run metrics and historical verification/failure records |
| [warning-detection/](warning-detection/) | Fresh assessment, per-worker outcomes, runtime audit and earlier-policy failures |
| [language-model/](language-model/) | Initial/final measurements, calibration comparisons and [final training audit proofs](language-model/final-training/) |
| [additional-studies/](additional-studies/) | Learning-rate and SVHN comparisons with adverse outcomes and scope limits |

Digit accuracy values in CSVs are fractions: `0.91968` means about `91.97%`.
Reported sample SD is across three seed-level means, not independent workers.
Ordinary training has no private epsilon; a blank value is not zero. Optional
studies use separate declared evaluation populations and do not replace the
original matched baseline.

The [retained logs](../audit/README.md) preserve original event records used to
check the numeric results. Files labelled derived or summary remain derived;
renaming them does not turn them into raw receipts. The language-model record
preserves its audited weight-handoff boundary and cumulative privacy history.
Assessment data remain unused.

Frozen settings are in [configurations/](../configurations/). Source identity and
availability limitations are recorded in [provenance/](../provenance/) and
[artifact-availability.json](../artifact-availability.json). Full model/CSPRNG
re-execution is not possible from this compact folder alone.
