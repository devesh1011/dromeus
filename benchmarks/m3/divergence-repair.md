# Divergence warning repair workflow

Canonical plan: [Obsidian](</Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/plans/M3 divergence warning repair plan.md>).

The original warning threshold set failed held-out assessment. The growth-ratio3 candidate is historical development evidence only; it is not a validated default. Core detector semantics, private training and historical evidence are unchanged.

## Reproduce the failure

Run the portable measured fixture (no large archives required):

```sh
uv run --frozen --no-sync python -m benchmarks.m3.repair_replay \
  --fixture tests/fixtures/divergence_repair/historical.json
```

This exits1 intentionally: the three minimized monitor traces contain two false entries and one missed new fault warning. Fixture hashes and original log/manifest identities are retained. The traces are truncated regressions, not final assessment.

Run the separately labelled development probe:

```sh
uv run --frozen --no-sync python -m benchmarks.m3.repair_replay \
  --fixture tests/fixtures/divergence_repair/historical.json --probe growth_ratio=3
```

The candidate removes the fixture false alarms and detects both faults at delay2. Full archived15-group replay also changes32 false entries/four misses to zero/zero, preserving two-round detected delay. These old seeds are development data. Full replay commands, per-round diagnostics and all original live metrics are retained under `benchmarks/results/m3/divergence-repair-2026-10-25/historical-replay/`.

## Prepare a fresh campaign

The successor campaign tool preserves the historical seed-specific loader. It requires a separate predeclaration with exact required run recipes, fresh disjoint seed sets, public participant keys, explicit finite policy candidates and engineering criteria.

```sh
uv run --frozen --no-sync python -m benchmarks.m3.detector_campaign template \
  --output /tmp/dromeus-warning-campaign-template.json
```

The output is a schema and instructions, deliberately not a launchable campaign. Fill actual measured recipe/source/peer metadata before freezing:

```sh
uv run --frozen --no-sync python -m benchmarks.m3.detector_campaign freeze-spec \
  --spec campaign.json --output predeclared.json
uv run --frozen --no-sync python -m benchmarks.m3.detector_campaign calibrate \
  --predeclared predeclared.json --index calibration-index.json --output frozen.json
uv run --frozen --no-sync python -m benchmarks.m3.detector_campaign evaluate \
  --predeclared predeclared.json --frozen frozen.json --index validation-index.json \
  --require-live --runtime-root collected-campaign --output assessment.json
```

For readiness assessment, the predeclared spec must also contain the actual runtime source hashes and execution environment in `runtime_contract`. Retain full worker artifacts under `collected-campaign/runs/<run-id>/rank-N/`. The tool reuses the existing full run auditor to check committed completion, logical steps, injected faults, accounting, model exports and distinct-worker/runtime provenance, then binds those artifacts to the exact assessed logs. Missing proof leaves readiness false; `--require-live` rejects it.

Freeze before collecting calibration evidence. Calibration opens only its declared calibration evidence. Assessment binds the chosen policy to live sealed manifests and terminal zero-drop summaries. It rejects missing/altered/reused evidence, checks live/replay source decisions and reports live detection delays separately from canonical replay. A pre-active warning never becomes a new fault detection. Supplemental warning occupancy does not replace primary entry scoring.

This first campaign tool validates the existing manifestv5 digit recipe. It does not imply language-model threshold validation or broaden private model profiles. A small passing suite establishes only its declared workload scope.

## Verification and execution status

Focused regression/runtime checks, full verification and real local AXL results are recorded in the new repair evidence tree. No new GPU campaign is started by these tools. Fresh calibration/held-out assessment require a frozen execution/resource plan and an explicitly authorized compute ceiling. Historical evidence and intentionally uncommitted pre-existing benchmark dependencies remain local; this workflow does not silently publish them.
