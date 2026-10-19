# Privacy-budget forecasting

Run prospective accounting locally before selecting a training horizon or spending
on GPU pilots. This tool performs no training, sampling, reservation or cloud action.
It uses the same pinned Opacus 1.6.0 PRV accountant and error settings as Dromeus's
runtime, with an independent RDP result for comparison. PRV remains the primary
bound; the tool never chooses the smaller estimate or falls back on accountant
failure.

```bash
uv run --no-sync python -m benchmarks.m3.forecast_cli \
  --config benchmarks/m3/configs/privacy-pilot.json \
  --output /path/to/new-forecast.json \
  --timeout-seconds 120
```

Install the optional privacy dependencies first (`uv sync --frozen --extra privacy`).
The CPU forecast does not require CUDA, CSPRNG, training tensors or an optimizer.
The example has four public digit sources, 3,000 records/node, requested batch64,
sigma candidates 0.5/1/2/4, horizons 250/750/1,500 logical steps, three repetitions
and delta=1e-5. These are illustrative pilot candidates from the canonical guide,
**not a frozen experiment or an adopted privacy target**. The previous non-private
public-digit releases are outside this hypothetical DP mechanism forecast; an
empty prior list cannot erase an earlier non-private release of protected data.

## Inputs

`ForecastPlan` is also available from `benchmarks.m3.privacy_forecast` for Python
callers. Use `parse_plan(config_bytes)` for config files (it rejects duplicate
JSON keys recursively) and
`forecast(plan)` for synchronous calculation. The CLI isolates that call in a
child process with a deadline; direct Python callers own their execution limits.

- `nodes`: unique opaque node IDs, fixed `sample_count`, and **required**
  `prior_history` entries with `noise_multiplier`, `sample_rate`, `logical_steps`.
  Explicit `[]` means the input assumes no prior mechanisms to compose.
- `batch_size`: shared requested batch size, matching the owned fixed-tensor
  loader with `drop_last=False`. Nodes may have unequal sample counts.
- `noise_multipliers`: positive distinct sigma candidates. Zero-noise and ordinary
  baseline training are non-private and do not belong in this grid.
- `logical_step_horizons`: positive distinct counts of local optimizer updates.
  Convert outer rounds using `round_count * local_steps`. Physical chunks,
  retransmissions and accountant history segments are not additional steps.
- `repetitions`: number of runs of each candidate on the same records, including
  repetitions with different research seeds.
- `delta` and `delta_rationale`: one fixed reporting delta and the reason for its
  selection. The report includes `delta_times_sample_count` for each node. No
  population-relative cutoff or default delta is imposed by this tool.
- `randomness_profile`: `secure` or `reproducible_public_benchmark`, copied to the
  input provenance. Selecting secure does not verify secure RNG. Seeded public
  experiments remain mathematical mechanism estimates.
- `epsilon_limit`: optional positive cap, evaluated against PRV for each displayed
  history; omitted/null reports no pass/fail decision.
- `prv_epsilon_error` and `prv_delta_error_ratio`: defaults 0.01 and 0.001, matching
  the runtime. Effective PRV delta error is `delta * prv_delta_error_ratio`.

Use prior **reserved horizons**, including failed/aborted runs, and include any
external exposures of that data lineage. Preserve each historical rate and sigma;
do not recalculate them using the new dataset size. This version accepts explicit
history snapshots and does not read or modify live ledgers. Live preflight and
reservation enforcement remain authoritative and must recheck the full history
at execution time. Privacy accounting cannot certify input completeness.

For example, prior history may contain:

```json
{
  "node_id": "node-0",
  "sample_count": 3000,
  "prior_history": [
    {"noise_multiplier": 1.0, "sample_rate": 0.025, "logical_steps": 500},
    {"noise_multiplier": 2.0, "sample_rate": 0.02, "logical_steps": 750}
  ]
}
```

## Interpreting the report

For the pinned loader, `q = 1 / ceil(N / requested_batch_size)`. At N=3,000 and
batch64 this is **1/47**, not 64/3,000. The integer gradient normalization divisor
is `int(N*q) = 63`. Expected sampled examples use the unrounded N*q, so the report
keeps them distinct. Empty draws still count as logical mechanisms.

Each node includes:

- `prior`: the supplied history alone.
- `one_run`: one candidate run, excluding prior spending.
- `repetitions_only`: all repetitions of that candidate, excluding prior spending.
- `prior_plus_repetitions`: the prospective total if that candidate is selected.
- `prior_plus_entire_sweep`: the total if **every** sigma/horizon combination and
  repetition is executed on that node's records.

Each estimate retains its composed mechanism history, total logical-step count,
PRV epsilon, RDP epsilon, numerical warnings and cap result. Composition operates on histories; it
never adds per-run epsilon values or subtracts a prior epsilon to manufacture a
remaining budget. Candidate alternatives do not spend budget until run, but
executing all of them accumulates the full sweep. Repeating clipping trials at
fixed sigma also consumes additional mechanisms; include those repetitions or
histories even though clipping C itself cancels from accounting at fixed sigma.
Optimizer choice also does not change this accountant calculation when sampling,
noise and logical steps match.

`candidate_maxima` and `max_local_entire_sweep_prv_epsilon` are descriptive
worst-local values. **`group_privacy_bound` is deliberately null.** Different dataset
names do not establish record/person disjointness; a whole-transcript bound needs
the separately verified ownership and post-processing assumptions in the canonical
M3 privacy contract. The tool does not average local epsilons.

Zero epsilon for an empty prior history means no supplied spending. The ordinary
non-private baseline must be shown separately, never plotted at epsilon zero.
Forecasts are prospective budgets, not achieved accountant histories or accuracy
results. Exact logical work counts are reported; GPU cost, time and memory require
measurements of the actual workload and hardware.

## Reproducibility and failure behavior

Output includes the normalized input and canonical hash, exact config-file SHA-256,
source hashes, package versions, accountant settings and UTC generation time.
Existing files and input aliases are never overwritten. Complete output is staged,
flushed and published atomically. Invalid inputs, accountant exceptions/nonfinite
results, child failures and timeouts exit nonzero without publishing a completed
report. A timeout defaults to120 seconds and may be set up to3,600 seconds.

Input bounds limit nodes, candidate count, history length and file size. PRV can
still consume substantial memory for extreme noise/horizon/tolerance choices;
a deadline is not a memory quota. Start with the provided modest pilot range.
On failure, adjust the explicitly declared candidate grid or numerical settings
and retain those changes with the next forecast. Never report failed estimates as
zero or as passing the cap.

```bash
uv run --no-sync pytest tests/benchmarks/test_privacy_forecast.py \
  tests/benchmarks/test_forecast_cli.py -q
```

Tests compare retained W0 and independent accountant results, actual loader
rounding, unequal nodes, historical/repeated/full-sweep composition, cap semantics,
invalid input, numerical failures and CLI provenance/deadline/no-overwrite behavior.
