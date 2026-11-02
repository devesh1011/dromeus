# M3 proposal check

## Basis and overall result

The available proposal defines M3 as **DP-SGD and convergence hardening** and
states the deliverable as:

> tunable DP-SGD working end to end, published tradeoff curves, divergence detection.

The [proposal basis](proposal-basis.json) retains the full M3 paragraph and the
source note's SHA-256. The canonical implementation guide treats M3 as the
original proposal scope. A separate signed agreement was not located in the
scoped repository, Dromeus vault or focused connected-source search. This is a
check of the available proposal, not verification of an unseen agreement or
amendment.

**Verdict:** retained evidence supports the technical deliverables within their
declared profiles. Publication of the tradeoff curves and Gensyn acceptance
remain unverified. Do not label the contractual milestone fully accepted on the
basis of local preparation alone.

## Requirement map

| Proposal requirement | Verdict | Direct evidence | Limits |
| --- | --- | --- | --- |
| Integrate Opacus DP-SGD and make noise multiplier, clipping bound and related hyperparameters configurable | Supported for the declared profiles | [Execution summary](results/private-training/implementation-and-execution-summary.json); [frozen recipes and run matrix](configurations/frozen-experiment.json); [secure configuration](configurations/secure-run.json); current source and checks below | Trusted application and supported model/optimizer contracts; secure CSPRNG compatibility is profile-specific. Secure crash resume and arbitrary callbacks are not certified. |
| Tunable DP-SGD working end to end | Supported by recorded real-AXL execution | Separate secure digit run: 750 updates on each of four workers; generic secure four-worker conformance summarized in the [execution record](results/private-training/implementation-and-execution-summary.json) | Historical retained metrics/report derivatives are not a fresh remote rerun or a complete raw receipt bundle. Fractional GPU declarations in the original digit campaign do not prove four physical chips. |
| Published utility-versus-privacy curves so a group can select its setting from measurements | Measurements supported; publication unverified | [Five-setting utility table](results/privacy-utility/accuracy-vs-privacy.csv), [three-seed settings](configurations/privacy-utility-settings.json), [node data](results/privacy-utility/worker-results.csv), [trajectories](results/privacy-utility/learning-trajectory.csv); charts in the linked report | Three repeats, matched ordinary baseline and four positive noise settings. Public seeded values are per-run, per-worker mechanism estimates at delta 10^-5. A private document or local folder is not evidence of public publication. |
| Harden consensus-distance tracking into divergence detection that flags instability for operator retuning | Supported for the frozen digit workload | [Fresh held-out assessment](results/warning-detection/assessment.json), [runtime audit](results/warning-detection/runtime-audit.json), [deployed policy](configurations/divergence-policy.json) | Zero false-warning groups among 11 without injected faults; all three fault groups detected by all monitors after two completed exchange rounds. This does not establish universal thresholds, text-warning readiness or automatic retuning. |

## Numbers a reviewer can cross-check

The ordinary digit baseline averaged **91.97%** accuracy. Private settings
sigma `0.5 / 1 / 2 / 4` achieved approximately `80.70 / 73.58 / 63.64 / 54.62%`,
with primary per-worker epsilon `22.3286 / 3.6203 / 1.2310 / 0.5410` at delta
`10^-5`. The CSVs retain full precision and seed-level variation.

The warning repair used 12 calibration groups followed by 14 fresh assessment
groups (56 worker executions). All 56 runtime audits passed; recorded event drops
were zero and live/replay decisions agreed. The [original adverse results](results/warning-detection/earlier-policy-failures.json)
remain evidence: the original main utility cohort had false warnings in 5/10
groups, and not all injected-fault monitors produced a new warning. The later
policy does not erase those failures.

The [retained log index](audit/log-index.json) links original-event streams for all
88 accepted digit workers and all 56 warning-assessment workers, with calibration
streams listed separately. These preserve round metrics, evaluations, consensus
observations, warning transitions and completion records. The
[portable log verifier](verification/verify-retained-logs.py) checks their hashes,
round coverage and agreement with the digit result tables. Complete verbose logs
were retired; routine successful per-message transport records were filtered out.

The additional GPT campaign finished 8,192 updates per worker across its recorded
execution boundary, with final full-development loss **7.198572** and mean
worker perplexity **1,337.519**. Model, ancestry, accounting, delivery and bounded
resource checks passed. Assessment remained unused; this is not a general
language-quality or text-divergence guarantee.

## Implementation and current-source verification

Relevant maintained modules are:

- [Manifest privacy policy](../../../src/dromeus/manifests/models.py)
- [Application preparation](../../../src/dromeus/application.py)
- [Private trainer](../../../src/dromeus/training/private_trainer.py)
- [Privacy mechanism/accounting](../../../src/dromeus/training/privacy.py)
- [Persistent privacy reservations](../../../src/dromeus/persistence/privacy_ledger.py)
- [NoLoCo update path](../../../src/dromeus/algorithms/noloco.py)
- [Divergence state machine](../../../src/dromeus/telemetry/divergence.py)
- [Runtime divergence monitor](../../../src/dromeus/telemetry/divergence_monitor.py)

See [local verification](verification/README.md) for the exact commands, source
identity and results from this preparation. The checkout contains pre-existing
uncommitted work. Historical experiment-source hashes and historical test totals
do not certify every current change. No production-source changes were made for
this handoff.

The focused current-checkout check passed **85 tests with no skips or failures**;
the architecture check also passed. Source and selected-test hashes stayed
unchanged during those checks. This was a local CPU check, not a fresh GPU/AXL
campaign or a full-suite rerun.

## Remaining submission checks

1. Publish or otherwise provide the tradeoff curves through the channel required
   by the team, and record the resulting accessible URL. The charts need not be
   duplicated in this evidence folder.
2. Compare any separately signed agreement or amendments with this proposal map
   if those documents impose additional conditions.
3. Record the team's milestone review and acceptance. Local tests and prepared
   evidence are not acceptance by Gensyn.

The proposal does not set a minimum GPT accuracy, maximum perplexity, maximum
allowed accuracy loss, fixed epsilon target or a universal warning rate for M3.
The $2,000 and approximately four weeks are the proposal's milestone allocation
and estimate. The broader library release, external user and public multi-region
dashboard belong to M4. Internal pilot choices and optional later studies are
not additional grant criteria.
