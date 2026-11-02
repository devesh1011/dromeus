# Current checkout: focused local verification

This is a new local CPU check of the dirty checkout, separate from historical frozen experiment receipts. It does not establish a new real-AXL/GPU run or milestone acceptance.

Latest result: [local-20261102T103000Z-result.json](local-20261102T103000Z-result.json).

- Git HEAD: `08a513375bc40cc44dcd4836da03dc270a7aca92`
- Result: **PASS_BOUNDED_CURRENT_CHECKOUT_CHECKS**
- Tests: 85 passed; 0 skipped; 0 expected failures; 0 failed; 0 errors.
- Architecture check exit: 0.
- Runtime source and selected test hashes unchanged during checks: True.

## Exact commands

```sh
.venv/bin/python -m pytest -q -ra -p no:cacheprovider --basetemp benchmarks/results/m3/verification/local-20261102T103000Z-temporary --junitxml benchmarks/results/m3/verification/local-20261102T103000Z-temporary.xml tests/unit/manifests/test_privacy_policy.py tests/unit/training/test_private_trainer.py tests/unit/training/test_privacy_query.py tests/unit/persistence/test_privacy_ledger.py tests/unit/runtime/test_private_application.py tests/unit/telemetry/test_divergence.py tests/unit/telemetry/test_divergence_monitor.py tests/unit/runtime/test_divergence_runtime.py
.venv/bin/python scripts/check_structure.py
```

The temporary JUnit and synthetic test directories were removed after extracting counts. Logs retain the test output with absolute home/repository prefixes sanitized. The JSON records package versions, exact commands, exit codes, skipped checks, and before/after source/test hashes.

## Scope

The tests cover configurable privacy policy binding, analytic clipping and direct Opacus parity for supported optimizers, logical/microbatch and empty-draw accounting, PRV/RDP queries, durable lineage reservations, private application admission, detector state transitions, and nonblocking warning delivery/runtime composition. Secure randomness tests may be skipped when the local CSPRNG dependency is unavailable; see the exact reasons in the result JSON. Real AXL, GPU execution, the full suite, and publication/acceptance are outside this check.

## Current implementation map

These anchors identify the inspected checkout, not a claim that its dirty files are identical to a historical experiment archive.

| Required feature | Current source anchors | Focused test anchors |
| --- | --- | --- |
| Configurable clipping/noise and policy binding | `src/dromeus/manifests/models.py:417`; `src/dromeus/training/private_trainer.py:160` | `tests/unit/manifests/test_privacy_policy.py:10`; `tests/unit/training/test_private_trainer.py:206` |
| Shared Opacus mechanism with SGD/Adam/AdamW | `src/dromeus/training/private_trainer.py:164`; retained wrapped optimizer and parameter-copy lifecycle | `tests/unit/training/test_private_trainer.py:125` direct Opacus trajectory parity; `:529` equal accounting |
| Actual sampler rate, logical steps, PRV/RDP | `src/dromeus/training/private_trainer.py:186`, `:323`, `:361`; `src/dromeus/training/privacy.py:39` | `tests/unit/training/test_private_trainer.py:296`, `:359`, `:591`; `tests/unit/training/test_privacy_query.py:49` |
| Durable full-horizon composition and failure handling | `src/dromeus/training/privacy.py:81`; `src/dromeus/persistence/privacy_ledger.py:95` | `tests/unit/persistence/test_privacy_ledger.py:11`, `:42`, `:67`; `tests/unit/training/test_private_trainer.py:337` |
| Application-to-NoLoCo/AXL wiring | `src/dromeus/application.py:102`, `:332`; `src/dromeus/runtime.py:576`; `src/dromeus/gossip/axl.py:123`; `src/dromeus/transport/axl.py:38` | `tests/unit/runtime/test_private_application.py:56`, `:159` run here; opt-in real-network `tests/integration/test_private_training_axl.py:40` was **not run** |
| Live divergence state machine and warning delivery | `src/dromeus/telemetry/divergence.py:55`; `src/dromeus/telemetry/divergence_monitor.py:73`; `src/dromeus/runtime.py:662`, `:1041`, `:1126` | `tests/unit/telemetry/test_divergence.py:42`, `:65`, `:95`, `:185`; `test_divergence_monitor.py:15`, `:67`; `tests/unit/runtime/test_divergence_runtime.py:63` |

The configured detector emits warnings and missing-data status; it does not automatically retune training. Local state-machine tests do not establish workload-specific false-warning rates or text-threshold readiness. Real-AXL implementation wiring is present and has an opt-in four-process optimizer integration test, but this check deliberately executed no real transport. The selected tests do not replace current full-suite, native-transformer, warm-start, or GPU validation.


## Retained event-log verification

Run `python3 verification/verify-retained-logs.py` from the M3 evidence root. This standard-library check writes nothing unless `--output` is supplied. The [saved receipt](retained-log-check.json) verifies 232 plain-text JSONL streams, 190,041 retained original event records and 2,552 numeric values against the shipped digit CSVs. It checks hashes, identities, 75-round coverage, five evaluations and completion for every worker. Accepted digit and warning runs retain modern consensus observations; original calibration logs use their recorded older distance-event schema. A deliberately wrong stored-file hash was also rejected.

These are filtered original records, not reconstructed logs or complete verbose originals. Full model/CSPRNG/transport replay and independent remote attestation are outside this check.
