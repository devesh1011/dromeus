"""Bounded deterministic warnings from ordered, complete consensus observations."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum
from statistics import median
from threading import RLock

from dromeus.manifests.models import DivergencePolicy
from dromeus.telemetry.consensus import ConsensusDistance, ConsensusObservation


class DetectorState(StrEnum):
    WARMING_UP = "warming_up"
    HEALTHY = "healthy"
    SUSPECT = "suspect"
    DIVERGING = "diverging"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class DetectorSnapshot:
    state: DetectorState = DetectorState.WARMING_UP
    warning_active: bool = False
    scale_warning: bool = False
    reason: str = "warmup"
    source_round: int | None = None
    detection_round: int = -1
    window_start: int | None = None
    window_end: int | None = None
    window_count: int = 0
    expected_participants: int = 0
    observed_participants: int = 0
    age_rounds: int = 0
    missing_rounds: int = 0
    measurement_channel: str = "normalized"
    earlier_median: float | None = None
    later_median: float | None = None
    log_slope: float | None = None
    normalized_rms: float | None = None
    absolute_rms_spread: float | None = None
    mean_sketch_norm: float | None = None


@dataclass(frozen=True, slots=True)
class DetectorTransition:
    sequence: int
    previous_state: DetectorState
    snapshot: DetectorSnapshot


class DivergenceDetector:
    """Thread-safe pure state machine. Observation and commit order may differ.

    A watermark is a locally committed round. Missing rounds older than the lag
    allowance finalize as gaps. Warnings survive gaps until valid recovery evidence.
    Work and memory are bounded by window/lag sizes, including watermark jumps.
    """

    def __init__(self, policy: DivergencePolicy, *, participant_count: int) -> None:
        if participant_count < 2:
            raise ValueError("detector requires at least two participants")
        self.policy = policy
        self._participants = participant_count
        self._lock = RLock()
        self._pending: dict[int, ConsensusDistance] = {}
        self._window: deque[ConsensusDistance] = deque(maxlen=policy.window_rounds)
        self._next = 0
        self._watermark = -1
        self._last_valid = -1
        self._growth = self._severe = self._recovery = 0
        self._sequence = 0
        self._ignored = 0
        self._invalid = 0
        self._snapshot = DetectorSnapshot(expected_participants=participant_count)

    @property
    def snapshot(self) -> DetectorSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._pending) + len(self._window)

    @property
    def ignored_observations(self) -> int:
        return self._ignored

    @property
    def invalid_observations(self) -> int:
        return self._invalid

    def observe(self, observation: ConsensusDistance) -> tuple[DetectorTransition, ...]:
        with self._lock:
            if (
                observation.sketch_count != self._participants
                or observation.round_id < 0
                or not math.isfinite(observation.normalized_rms)
                or observation.normalized_rms < 0
            ):
                self._invalid += 1
                return ()
            round_id = observation.round_id
            if (
                round_id < self._next
                or round_id > self._watermark + self.policy.max_lag_rounds + 1
            ):
                self._ignored += 1
                return ()
            existing = self._pending.get(round_id)
            if existing is not None:
                if existing != observation:
                    self._invalid += 1
                else:
                    self._ignored += 1
                return ()
            self._pending[round_id] = observation
            return self._drain()

    def advance(self, committed_round: int) -> tuple[DetectorTransition, ...]:
        if committed_round < 0:
            raise ValueError("committed round must be nonnegative")
        with self._lock:
            if committed_round <= self._watermark:
                return ()
            self._watermark = committed_round
            return self._drain()

    def _emit(self, candidate: DetectorSnapshot) -> tuple[DetectorTransition, ...]:
        old = self._snapshot
        self._snapshot = candidate
        if (old.state, old.warning_active, old.scale_warning) == (
            candidate.state,
            candidate.warning_active,
            candidate.scale_warning,
        ):
            return ()
        self._sequence += 1
        return (DetectorTransition(self._sequence, old.state, candidate),)

    def _drain(self) -> tuple[DetectorTransition, ...]:
        transitions: list[DetectorTransition] = []
        while self._next <= self._watermark:
            observation = self._pending.pop(self._next, None)
            if observation is not None:
                self._next += 1
                transitions.extend(self._consume(observation))
                continue
            cutoff = self._watermark - self.policy.max_lag_rounds
            if self._next >= cutoff:
                break
            next_pending = min(self._pending, default=cutoff)
            end = min(cutoff, next_pending)
            missing = end - self._next
            self._next = end
            self._window.clear()
            self._growth = self._severe = self._recovery = 0
            transitions.extend(
                self._emit(
                    replace(
                        self._snapshot,
                        state=DetectorState.INSUFFICIENT_DATA,
                        reason="missing_rounds",
                        detection_round=self._watermark,
                        window_start=None,
                        window_end=None,
                        window_count=0,
                        observed_participants=0,
                        missing_rounds=self._snapshot.missing_rounds + missing,
                        age_rounds=self._watermark - self._last_valid,
                        earlier_median=None,
                        later_median=None,
                        log_slope=None,
                    )
                )
            )
        self._snapshot = replace(
            self._snapshot,
            detection_round=self._watermark,
            age_rounds=max(0, self._watermark - self._last_valid),
        )
        return tuple(transitions)

    def _consume(self, point: ConsensusDistance) -> tuple[DetectorTransition, ...]:
        p = self.policy
        self._last_valid = point.round_id
        candidate = replace(
            self._snapshot,
            source_round=point.round_id,
            detection_round=self._watermark,
            age_rounds=max(0, self._watermark - point.round_id),
            observed_participants=point.sketch_count,
            normalized_rms=point.normalized_rms,
            absolute_rms_spread=point.absolute_rms_spread
            if isinstance(point, ConsensusObservation)
            else None,
            mean_sketch_norm=point.mean_sketch_norm
            if isinstance(point, ConsensusObservation)
            else None,
            scale_warning=point.denominator_degenerate
            if isinstance(point, ConsensusObservation)
            else False,
        )
        if point.round_id < p.warmup_rounds:
            return self._emit(replace(candidate, reason="warmup"))
        self._window.append(point)
        absolute = any(
            isinstance(x, ConsensusObservation) and x.denominator_degenerate
            for x in self._window
        )
        channel = "absolute" if absolute else "normalized"
        if channel != self._snapshot.measurement_channel:
            self._growth = self._severe = self._recovery = 0
        floor = p.absolute_distance_floor if absolute else p.distance_floor
        severe = p.absolute_severe_distance if absolute else p.severe_distance
        recovery = p.absolute_recovery_distance if absolute else p.recovery_distance
        candidate = replace(
            candidate,
            measurement_channel=channel,
            window_start=self._window[0].round_id,
            window_end=point.round_id,
            window_count=len(self._window),
            earlier_median=None,
            later_median=None,
            log_slope=None,
        )
        if (
            floor is None
            or severe is None
            or recovery is None
            or absolute
            and any(not isinstance(x, ConsensusObservation) for x in self._window)
        ):
            self._growth = self._severe = self._recovery = 0
            return self._emit(
                replace(
                    candidate,
                    state=DetectorState.INSUFFICIENT_DATA,
                    reason="denominator_degenerate",
                )
            )
        values = [
            x.absolute_rms_spread
            if absolute and isinstance(x, ConsensusObservation)
            else x.normalized_rms
            for x in self._window
        ]
        self._severe = self._severe + 1 if values[-1] > severe else 0
        full = len(values) == p.window_rounds
        growth = False
        if full:
            half = len(values) // 2
            early, late = median(values[:half]), median(values[half:])
            # Centered rounds avoid cancellation for large absolute round IDs.
            xs = [x.round_id - self._window[0].round_id for x in self._window]
            ys = [math.log(max(x, floor)) for x in values]
            xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
            slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys, strict=True)) / sum(
                (x - xm) ** 2 for x in xs
            )
            growth = (
                late > floor
                and late / max(early, floor) > p.growth_ratio
                and slope > p.min_log_slope
            )
            candidate = replace(
                candidate, earlier_median=early, later_median=late, log_slope=slope
            )
        self._growth = self._growth + 1 if growth else 0
        if self._growth >= p.patience_windows or self._severe >= p.severe_patience:
            self._recovery = 0
            return self._emit(
                replace(
                    candidate,
                    state=DetectorState.DIVERGING,
                    warning_active=True,
                    reason="severe_distance"
                    if self._severe >= p.severe_patience
                    else "sustained_growth",
                )
            )
        if candidate.warning_active and not full and candidate.missing_rounds:
            self._recovery = 0
            return self._emit(
                replace(
                    candidate,
                    state=DetectorState.INSUFFICIENT_DATA,
                    reason="collecting_window",
                )
            )
        if candidate.warning_active:
            self._recovery = (
                self._recovery + 1
                if full and max(values) < recovery and not growth
                else 0
            )
            if self._recovery >= p.recovery_windows:
                return self._emit(
                    replace(
                        candidate,
                        state=DetectorState.HEALTHY,
                        warning_active=False,
                        reason="recovered",
                    )
                )
            return self._emit(
                replace(
                    candidate, state=DetectorState.DIVERGING, reason="awaiting_recovery"
                )
            )
        if growth or self._severe:
            return self._emit(
                replace(
                    candidate,
                    state=DetectorState.SUSPECT,
                    reason="growth" if growth else "severe_distance",
                )
            )
        if full:
            return self._emit(
                replace(candidate, state=DetectorState.HEALTHY, reason="stable_window")
            )
        return self._emit(
            replace(
                candidate,
                state=DetectorState.WARMING_UP
                if candidate.missing_rounds == 0
                else DetectorState.INSUFFICIENT_DATA,
                reason="collecting_window",
            )
        )
