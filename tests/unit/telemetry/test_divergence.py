"""Behavioral detector seam; synthetic thresholds are not deployment defaults."""

from dromeus.manifests.models import DivergencePolicy
from dromeus.telemetry.consensus import ConsensusObservation
from dromeus.telemetry.divergence import (
    DetectorState,
    DetectorTransition,
    DivergenceDetector,
)


def policy(**changes: object) -> DivergencePolicy:
    values: dict[str, object] = dict(
        threshold_set_id="synthetic-test-v1",
        warmup_rounds=0,
        window_rounds=4,
        patience_windows=2,
        distance_floor=0.01,
        growth_ratio=1.5,
        min_log_slope=0.1,
        severe_distance=100.0,
        severe_patience=2,
        recovery_windows=2,
        recovery_distance=0.3,
        max_lag_rounds=1,
    )
    values.update(changes)
    return DivergencePolicy.model_validate(values)


def point(round_id: int, distance: float) -> ConsensusObservation:
    return ConsensusObservation(
        round_id=round_id,
        normalized_rms=distance,
        absolute_rms_spread=distance * (1 + 1e-12),
        mean_sketch_norm=1,
        denominator_degenerate=False,
        sketch_count=4,
    )


def test_plateau_is_healthy_without_repeated_transitions() -> None:
    detector = DivergenceDetector(policy(), participant_count=4)
    transitions: list[DetectorTransition] = []
    for round_id, value in enumerate([0.2, 0.21, 0.19, 0.2, 0.21, 0.2, 0.19, 0.2]):
        transitions.extend(detector.observe(point(round_id, value)))
        transitions.extend(detector.advance(round_id))
    assert detector.snapshot.state == DetectorState.HEALTHY
    assert [x.snapshot.state for x in transitions] == [DetectorState.HEALTHY]


def feed(
    detector: DivergenceDetector, values: list[float], start: int = 0
) -> list[DetectorState]:
    states: list[DetectorState] = []
    for round_id, value in enumerate(values, start):
        changes = (
            *detector.observe(point(round_id, value)),
            *detector.advance(round_id),
        )
        states.extend(x.snapshot.state for x in changes)
    return states


def test_growth_requires_patience_and_recovery_requires_full_low_windows() -> None:
    detector = DivergenceDetector(policy(), participant_count=4)
    assert feed(detector, [0.1, 0.2, 0.4, 0.8, 1.6]) == [
        DetectorState.SUSPECT,
        DetectorState.DIVERGING,
    ]
    assert detector.snapshot.warning_active
    assert detector.snapshot.window_start == 1 and detector.snapshot.window_end == 4
    early = detector.snapshot.earlier_median
    assert early is not None and abs(early - 0.3) < 1e-10
    feed(detector, [0.1] * 4, start=5)
    assert detector.snapshot.warning_active
    feed(detector, [0.1], start=9)
    assert detector.snapshot.state == DetectorState.HEALTHY
    assert not detector.snapshot.warning_active


def test_severe_spike_needs_persistence_and_warmup() -> None:
    detector = DivergenceDetector(
        policy(warmup_rounds=2, severe_distance=2.0), participant_count=4
    )
    feed(detector, [10, 10])
    assert detector.snapshot.state == DetectorState.WARMING_UP
    feed(detector, [10, 0.1], start=2)
    assert not detector.snapshot.warning_active
    feed(detector, [10, 10], start=4)
    assert detector.snapshot.warning_active
    assert detector.snapshot.reason == "severe_distance"


def test_reordering_duplicates_late_points_and_gaps() -> None:
    detector = DivergenceDetector(policy(max_lag_rounds=2), participant_count=4)
    detector.advance(2)
    assert not detector.observe(point(2, 0.4))
    assert not detector.observe(point(0, 0.1))
    detector.observe(point(1, 0.2))
    detector.observe(point(1, 999))
    feed(detector, [0.8, 1.6], start=3)
    assert detector.snapshot.warning_active
    assert detector.ignored_observations == 1
    detector.advance(9)
    assert detector.snapshot.state == DetectorState.INSUFFICIENT_DATA
    assert detector.snapshot.warning_active
    assert detector.snapshot.missing_rounds == 2
    assert not detector.observe(point(5, 0.1))
    feed(detector, [0.1] * 7, start=7)
    assert detector.snapshot.state == DetectorState.HEALTHY
    assert not detector.snapshot.warning_active


def test_incomplete_and_far_future_points_do_not_fill_windows() -> None:
    from dataclasses import replace

    detector = DivergenceDetector(policy(), participant_count=4)
    assert not detector.observe(replace(point(0, 0.1), sketch_count=3))
    assert not detector.observe(point(1000000, 0.1))
    detector.advance(1000000)
    assert detector.snapshot.state == DetectorState.INSUFFICIENT_DATA
    assert detector.buffered_count <= 6
    assert detector.invalid_observations == 1
    assert detector.snapshot.missing_rounds == 999999


def test_scale_collapse_is_distinct_from_divergence() -> None:
    detector = DivergenceDetector(policy(severe_distance=2.0), participant_count=4)
    for i in range(8):
        point_ = ConsensusObservation(
            round_id=i,
            normalized_rms=1e12,
            sketch_count=4,
            absolute_rms_spread=1,
            mean_sketch_norm=0,
            denominator_degenerate=True,
        )
        detector.observe(point_)
        detector.advance(i)
    assert detector.snapshot.scale_warning
    assert not detector.snapshot.warning_active
    assert detector.snapshot.reason == "denominator_degenerate"


def test_absolute_scale_can_support_divergence_when_configured() -> None:
    detector = DivergenceDetector(
        policy(
            absolute_distance_floor=0.01,
            absolute_severe_distance=10,
            absolute_recovery_distance=0.3,
        ),
        participant_count=4,
    )
    for i, value in enumerate([0.1, 0.2, 0.4, 0.8, 1.6]):
        point_ = ConsensusObservation(
            round_id=i,
            normalized_rms=value / 1e-12,
            sketch_count=4,
            absolute_rms_spread=value,
            mean_sketch_norm=0,
            denominator_degenerate=True,
        )
        detector.observe(point_)
        detector.advance(i)
    assert detector.snapshot.warning_active
    assert detector.snapshot.measurement_channel == "absolute"


def test_malformed_scale_is_rejected() -> None:
    from dataclasses import replace

    import pytest

    for field, value in [
        ("normalized_rms", float("nan")),
        ("mean_sketch_norm", -1),
        ("absolute_rms_spread", float("inf")),
        ("denominator_degenerate", True),
    ]:
        with pytest.raises(ValueError):
            replace(point(0, 0.2), **{field: value})


def test_window_and_pending_memory_stay_bounded() -> None:
    detector = DivergenceDetector(policy(max_lag_rounds=3), participant_count=4)
    feed(detector, [0.2] * 2000)
    assert detector.buffered_count == 4
    assert detector.advance(10**12)
    assert detector.buffered_count == 0


def test_legacy_replay_does_not_invent_scale_measurements() -> None:
    from dromeus.telemetry.consensus import ConsensusDistance

    detector = DivergenceDetector(policy(), participant_count=4)
    for i in range(5):
        detector.advance(i)
        detector.observe(
            ConsensusDistance(round_id=i, normalized_rms=0.2, sketch_count=4)
        )
    assert detector.snapshot.state == DetectorState.HEALTHY
    assert detector.snapshot.mean_sketch_norm is None
    assert detector.snapshot.absolute_rms_spread is None


def test_absolute_thresholds_are_complete_and_bounded() -> None:
    import pytest

    for values in (
        {"absolute_distance_floor": 0.1},
        {"window_rounds": 1000000},
        {"max_lag_rounds": 1000000},
        {
            "absolute_distance_floor": 0.1,
            "absolute_severe_distance": 0.5,
            "absolute_recovery_distance": 0.8,
        },
    ):
        with pytest.raises(ValueError):
            policy(**values)
