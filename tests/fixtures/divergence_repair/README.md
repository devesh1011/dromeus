# Public divergence repair regression traces

`historical.json` is a portable development fixture captured from the October 21
M3 public digit campaign through its strict scale-aware loader. It retains one
actual monitor from each of three runs: baseline seed 29 through the false entry
at round 19; fault seed 29 through round 27 (already active before injection at
25); and fault seed 41 through its correct detection at round 27. Prefixes begin
at round zero so the production detector's ordering and warm-up remain intact.

Every scale and round value is taken from the original loader observations. The
loader may algebraically recover group absolute spread from saved group
normalized spread and mean norm, as in the original assessment; this does not
invent per-node curves. Source log/manifest hashes, source observation counts,
original index/freeze hashes, policy hash, and fixture content hash are retained.
These truncated single-monitor traces cannot establish full-horizon recovery,
independent repetitions, live delivery, or held-out performance.

Regenerate from the retained public archive with:

```bash
uv run --no-sync python -m benchmarks.m3.repair_replay \
  --index benchmarks/results/m3/final-2026-10-21/detector/validation-index-retry1.json \
  --frozen benchmarks/results/m3/final-2026-10-21/detector/frozen-policy.json \
  --capture-fixture tests/fixtures/divergence_repair/historical.json
```

The command intentionally exits 1 because the original warning policy fails its
readiness check. An exit code of zero means the declared replay suite passes,
never production certification. `--probe growth_ratio=3` evaluates a development
candidate without modifying the fixture or original policy; it is not untouched
assessment. The unit test separately preserves the original failed result.
