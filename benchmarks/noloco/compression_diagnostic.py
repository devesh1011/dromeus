"""Fast deterministic differential loop for NoLoCo compression quality."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from dromeus.algorithms.base import UpdateBundle
from dromeus.algorithms.codec import (
    BitmapTopKInt8Codec,
    DenseInt8Codec,
    IdentityCodec,
    NamedSafetensorsUpdateBundleCodec,
    TopKInt8Codec,
    UpdateCodec,
)
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.gossip.peer_scheduler import PeerScheduler
from dromeus.manifests.models import AdamSettings, NoLoCoConfig, Tensor, TensorSchema

OuterCodec = Literal["identity", "topk", "bitmap"]
SlowCodec = Literal["identity", "dense"]


@dataclass
class _QuadraticTrainer:
    value: np.ndarray
    target: np.ndarray

    def train_local_steps(self, step_count: int) -> None:
        for _ in range(step_count):
            self.value += np.float32(0.01) * (self.target - self.value)

    def weights(self) -> dict[str, np.ndarray]:
        return {"weight": self.value.copy()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.value = weights["weight"].copy()

    @property
    def local_loss(self) -> float:
        return float(np.mean(np.square(self.value - self.target)))

    def evaluate(self) -> tuple[float, float]:
        return self.local_loss, 0.0


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    outer_codec: OuterCodec
    slow_codec: SlowCodec
    objective: float
    max_residual_to_signal_ratio: float


def run_diagnostic(
    *,
    outer_codec: OuterCodec,
    slow_codec: SlowCodec,
    top_k_fraction: float = 0.4,
    round_count: int = 20,
    tensor_size: int = 4096,
) -> DiagnosticResult:
    """Run one four-node codec candidate against a shared quadratic objective."""
    rng = np.random.default_rng(17)
    global_target = rng.normal(0, 1, tensor_size).astype(np.float32)
    targets = tuple(
        global_target + rng.normal(0, 0.2, tensor_size).astype(np.float32)
        for _ in range(4)
    )
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(tensor_size,)),)
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
    members = tuple(f"peer-{index}" for index in range(4))
    with tempfile.TemporaryDirectory(prefix="noloco-compression-diagnostic-") as root:
        algorithms: list[NoLoCoAlgorithm] = []
        for index, target in enumerate(targets):
            outer = _outer_codec(schema, outer_codec, top_k_fraction)
            slow = _slow_codec(schema, slow_codec)
            algorithms.append(
                NoLoCoAlgorithm(
                    trainer=_QuadraticTrainer(
                        value=np.zeros(tensor_size, dtype=np.float32),
                        target=target,
                    ),
                    tensor_schema=schema,
                    config=config,
                    artifact_codecs={
                        "outer_gradient": outer,
                        "slow_weights": slow,
                    },
                    manifest_codec_ids={
                        "outer_gradient": outer.codec_id,
                        "slow_weights": slow.codec_id,
                    },
                    bundle_codec=NamedSafetensorsUpdateBundleCodec(
                        artifact_root=Path(root) / str(index),
                        run_id="compression-diagnostic",
                        manifest_hash="0" * 64,
                        sender_public_key=members[index],
                        algorithm_id="noloco",
                        artifact_schemas={
                            "outer_gradient": _encoded_schema(outer, schema),
                            "slow_weights": _encoded_schema(slow, schema),
                        },
                    ),
                )
            )
        scheduler = PeerScheduler(
            members,
            seed=17,
            training_round_count=round_count,
            final_consensus_rounds=0,
        )
        maximum_ratio = 0.0
        for round_id in range(round_count):
            bundles: list[UpdateBundle] = []
            for algorithm in algorithms:
                algorithm.pre_local(round_id)
                algorithm.local_training()
                bundles.append(algorithm.post_local_bundle())
            for left, right in scheduler.schedule(round_id).pairs:
                left_index = members.index(left)
                right_index = members.index(right)
                left_update = algorithms[left_index].validate_peer(
                    bundles[right_index]
                )
                right_update = algorithms[right_index].validate_peer(
                    bundles[left_index]
                )
                algorithms[left_index].peer_apply(left_update)
                algorithms[right_index].peer_apply(right_update)
            for algorithm, bundle in zip(algorithms, bundles, strict=True):
                observations = algorithm.observations()
                maximum_ratio = max(
                    maximum_ratio,
                    observations.error_feedback_residual_to_signal_ratio or 0.0,
                )
                algorithm.release_bundle(bundle)
        values = tuple(
            algorithm.snapshot().weights["weight"] for algorithm in algorithms
        )
    objective = float(
        np.mean([np.mean(np.square(value - global_target)) for value in values])
    )
    return DiagnosticResult(
        outer_codec=outer_codec,
        slow_codec=slow_codec,
        objective=objective,
        max_residual_to_signal_ratio=maximum_ratio,
    )


def _outer_codec(
    schema: TensorSchema,
    codec: OuterCodec,
    fraction: float,
) -> UpdateCodec:
    if codec == "topk":
        return TopKInt8Codec(schema, top_k_fraction=fraction)
    if codec == "bitmap":
        return BitmapTopKInt8Codec(schema, top_k_fraction=fraction)
    return IdentityCodec("identity-v1")


def _slow_codec(schema: TensorSchema, codec: SlowCodec) -> UpdateCodec:
    if codec == "dense":
        return DenseInt8Codec(schema)
    return IdentityCodec("identity-v1")


def _encoded_schema(codec: UpdateCodec, logical_schema: TensorSchema) -> TensorSchema:
    return codec.encoded_schema_for(logical_schema)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outer",
        choices=("identity", "topk", "bitmap"),
        default="bitmap",
    )
    parser.add_argument("--slow", choices=("identity", "dense"), default="dense")
    parser.add_argument("--top-k-fraction", type=float, default=0.4)
    parser.add_argument("--round-count", type=int, default=20)
    parser.add_argument("--tensor-size", type=int, default=4096)
    parser.add_argument("--max-objective-ratio", type=float, default=1.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    baseline = run_diagnostic(
        outer_codec="identity",
        slow_codec="identity",
        round_count=arguments.round_count,
        tensor_size=arguments.tensor_size,
    )
    candidate = run_diagnostic(
        outer_codec=cast(OuterCodec, arguments.outer),
        slow_codec=cast(SlowCodec, arguments.slow),
        top_k_fraction=arguments.top_k_fraction,
        round_count=arguments.round_count,
        tensor_size=arguments.tensor_size,
    )
    ratio = candidate.objective / baseline.objective
    print(
        json.dumps(
            {
                "baseline_objective": baseline.objective,
                "candidate_objective": candidate.objective,
                "max_residual_to_signal_ratio": (
                    candidate.max_residual_to_signal_ratio
                ),
                "objective_ratio": ratio,
                "outer_codec": candidate.outer_codec,
                "slow_codec": candidate.slow_codec,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0 if ratio <= arguments.max_objective_ratio else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DiagnosticResult", "run_diagnostic"]
