from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.noloco.analysis import (
    AnalysisCheckpointCapturingAlgorithm,
    AnalysisCheckpointContext,
    AnalysisCheckpointWriter,
    analyze_analysis_checkpoints,
    required_analysis_storage_bytes,
    validate_analysis_storage,
)
from dromeus.algorithms.base import UpdateBundle
from dromeus.algorithms.codec import NamedSafetensorsUpdateBundleCodec
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.manifests.models import (
    AdamSettings,
    NoLoCoConfig,
    Tensor,
    TensorSchema,
    WarmupCosineSchedule,
)


class _Trainer:
    def __init__(self, value: float) -> None:
        self._weights = {"weight": np.array([value], dtype=np.float32)}

    def train_local_steps(self, step_count: int) -> None:
        assert step_count == 50
        self._weights["weight"] *= np.float32(0.9)

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}

    @property
    def local_loss(self) -> float:
        return float(np.square(self._weights["weight"][0]))

    def evaluate(self) -> tuple[float, float]:
        return self.local_loss, 0.0


def _context(*, rank: int, node_id: str) -> AnalysisCheckpointContext:
    return AnalysisCheckpointContext(
        experiment_sha256="1" * 64,
        run_config_sha256="2" * 64,
        initial_checkpoint_sha256="3" * 64,
        tensor_schema_hash="4" * 64,
        world_size=4,
        benchmark_seed=17,
        rank=rank,
        node_id=node_id,
        total_outer_steps=20,
        interval_rounds=5,
        storage_budget_bytes=256 * 1024 * 1024,
    )


def test_exact_analysis_reports_declining_source_weight_std_and_deletes_tensors(
    tmp_path: Path,
) -> None:
    members = tuple(f"key-{rank}" for rank in range(4))
    for rank, member in enumerate(members):
        writer = AnalysisCheckpointWriter(
            root=tmp_path / "analysis",
            context=_context(rank=rank, node_id=member),
        )
        for completed, spread in ((5, 4.0), (10, 3.0), (15, 2.0), (20, 1.0)):
            offset = np.float32((rank - 1.5) * spread)
            writer.write(
                completed_outer_steps=completed,
                slow_weights={
                    "weight": np.array([10.0 + offset], dtype=np.float32),
                    "bias": np.array([2.0], dtype=np.float32),
                },
            )

    report_path = tmp_path / "report.json"
    report = analyze_analysis_checkpoints(
        root=tmp_path / "analysis",
        report_path=report_path,
        learning_rate_schedule=WarmupCosineSchedule(
            schedule_id="linear-warmup-cosine-v1",
            total_inner_steps=1000,
            warmup_inner_steps=100,
            start_learning_rate=0.0001,
            peak_learning_rate=0.001,
            final_learning_rate=0.0001,
        ),
        inner_steps_per_round=50,
    )

    assert report.passed
    assert [point.completed_outer_steps for point in report.points] == [5, 10, 15, 20]
    assert report.points[0].weight_std_l2 > report.points[-1].weight_std_l2
    assert report.trend_slope_per_inner_step is not None
    assert report.trend_slope_per_inner_step < 0
    assert report.final_below_post_warmup_peak
    assert len(report.points[0].snapshot_sha256_by_rank) == 4
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
    assert not tuple((tmp_path / "analysis").rglob("*.safetensors"))


def test_analysis_writer_uses_only_frozen_cadence(tmp_path: Path) -> None:
    writer = AnalysisCheckpointWriter(
        root=tmp_path,
        context=_context(rank=0, node_id="key-0"),
    )

    assert not writer.should_write(completed_outer_steps=1)
    assert writer.should_write(completed_outer_steps=5)
    assert writer.should_write(completed_outer_steps=20)
    with pytest.raises(ValueError, match="cadence"):
        writer.write(
            completed_outer_steps=4,
            slow_weights={"weight": np.ones(1, dtype=np.float32)},
        )


def test_analysis_storage_preflight_is_scale_aware() -> None:
    required = required_analysis_storage_bytes(
        raw_parameter_bytes=44_695_848,
        total_outer_steps=20,
        interval_rounds=5,
    )

    assert required == 4 * (44_695_848 + 64 * 1024)
    validate_analysis_storage(
        required_bytes=required,
        budget_bytes=256 * 1024 * 1024,
        available_bytes=required,
    )
    with pytest.raises(ValueError, match="budget"):
        validate_analysis_storage(
            required_bytes=required,
            budget_bytes=required - 1,
            available_bytes=required,
        )
    with pytest.raises(ValueError, match="available"):
        validate_analysis_storage(
            required_bytes=required,
            budget_bytes=256 * 1024 * 1024,
            available_bytes=required - 1,
        )


def test_analysis_wrapper_captures_only_post_outer_frozen_cadence(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
    )
    config = NoLoCoConfig(
        alpha=0.5,
        beta=0.7,
        gamma=0.7,
        inner_steps=50,
        adam=AdamSettings(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_clip_norm=1.0,
        ),
    )
    algorithms: list[AnalysisCheckpointCapturingAlgorithm] = []
    for rank in range(2):
        delegate = NoLoCoAlgorithm(
            trainer=_Trainer(float(rank + 1)),
            tensor_schema=schema,
            config=config,
            bundle_codec=NamedSafetensorsUpdateBundleCodec(
                artifact_root=tmp_path / "bundles" / str(rank),
                run_id="analysis-test",
                manifest_hash="0" * 64,
                sender_public_key=f"key-{rank}",
                algorithm_id="noloco",
                artifact_schemas={
                    "outer_gradient": schema,
                    "slow_weights": schema,
                },
            ),
        )
        algorithms.append(
            AnalysisCheckpointCapturingAlgorithm(
                delegate=delegate,
                writer=AnalysisCheckpointWriter(
                    root=tmp_path / "analysis",
                    context=_context(rank=rank, node_id=f"key-{rank}"),
                ),
            )
        )
    for round_id in range(5):
        bundles: list[UpdateBundle] = []
        for algorithm in algorithms:
            algorithm.pre_local(round_id)
            algorithm.local_training()
            bundles.append(algorithm.post_local_bundle())
        left = algorithms[0].validate_peer(bundles[1])
        right = algorithms[1].validate_peer(bundles[0])
        algorithms[0].peer_apply(left)
        algorithms[1].peer_apply(right)
        for algorithm, bundle in zip(algorithms, bundles, strict=True):
            algorithm.release_bundle(bundle)

    assert len(tuple((tmp_path / "analysis").rglob("*.safetensors"))) == 2
    for rank in range(2):
        record = json.loads(
            (
                tmp_path / "analysis" / f"rank-{rank}" / "analysis.jsonl"
            ).read_text(encoding="utf-8")
        )
        assert record["completed_outer_steps"] == 5
