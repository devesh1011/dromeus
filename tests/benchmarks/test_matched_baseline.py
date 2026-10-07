"""Matched ordinary training at the benchmark trainer/application seams."""

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest
import torch
from torch import nn

from benchmarks.m3.matched_baseline import BaselineRecipe, MatchedBaseline
from dromeus.manifests.models import DraftRunSpec

pytest.importorskip("opacus")


def public_model() -> nn.Linear:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    return model


def test_baseline_uses_unclipped_expected_batch_gradient() -> None:
    baseline = MatchedBaseline(
        model_factory=public_model,
        optimizer_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
        inputs=torch.tensor([[1.0], [2.0]]),
        targets=torch.zeros(2, 1),
        recipe=BaselineRecipe(
            batch_size=2,
            logical_steps=2,
            loss="mse",
            optimizer_definition="torch.SGD:lr0.1;no-weight-decay",
        ),
        sampling_seed=17,
        local_data_identity="public-two-records-v1",
        max_physical_batch_size=1,
    )
    baseline.trainer.train_local_steps(1)
    # Full-batch MSE gradient at weight 1 is 5; ordinary SGD gives 0.5.
    np.testing.assert_allclose(baseline.trainer.weights()["weight"], [[0.5]], atol=1e-7)
    assert baseline.sample_rate == 1
    assert baseline.expected_batch_size == 2
    assert baseline.trainer.completed_steps == 1


@pytest.mark.parametrize("loss_kind", ["mse", "cross_entropy"])
def test_poisson_trajectory_matches_unclipped_zero_noise_opacus(
    loss_kind: Literal["mse", "cross_entropy"],
) -> None:
    from torch.utils.data import DataLoader, TensorDataset

    from dromeus.training.privacy import opacus_module

    x = torch.arange(1, 12, dtype=torch.float32).reshape(-1, 1) / 10
    y = x * 0.2 if loss_kind == "mse" else torch.arange(11) % 2

    def classifier() -> nn.Linear:
        model = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        return model

    factory = public_model if loss_kind == "mse" else classifier
    recipe = BaselineRecipe(
        batch_size=4,
        logical_steps=4,
        loss=loss_kind,
        schedule="linear_decay",
        optimizer_definition="torch.Adam:lr0.01",
    )
    baseline = MatchedBaseline(
        model_factory=factory,
        optimizer_factory=lambda m: torch.optim.Adam(
            m.parameters(), lr=0.01, foreach=False, fused=False
        ),
        inputs=x,
        targets=y,
        recipe=recipe,
        sampling_seed=17,
        local_data_identity="public-eleven",
        max_physical_batch_size=2,
    )
    model = factory()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, foreach=False, fused=False
    )
    ordinary = DataLoader(
        TensorDataset(x, y), batch_size=4, generator=torch.Generator().manual_seed(17)
    )
    engine = opacus_module().PrivacyEngine(accountant="prv", secure_mode=False)
    wrapped, dp_optimizer, loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=ordinary,
        noise_multiplier=0,
        max_grad_norm=1e6,
    )
    manager = opacus_module("opacus.utils.batch_memory_manager").BatchMemoryManager
    for step in range(4):
        with manager(
            data_loader=loader, max_physical_batch_size=2, optimizer=dp_optimizer
        ) as physical:
            for inputs, targets in physical:
                dp_optimizer.zero_grad()
                prediction = wrapped(inputs)
                if not len(inputs):
                    loss = prediction.sum()
                elif loss_kind == "mse":
                    loss = (prediction - targets).square().mean()
                else:
                    loss = nn.functional.cross_entropy(prediction, targets)
                cast(Any, loss).backward()
                dp_optimizer.step()
                if sum(n for _, _, n in engine.accountant.history) == step + 1:
                    break
        optimizer.param_groups[0]["lr"] = 0.01 * (1 - (step + 1) / 4)
        baseline.trainer.train_local_steps(1)
        np.testing.assert_allclose(
            baseline.trainer.weights()["weight"],
            model.weight.detach().numpy(),
            atol=1e-7,
            rtol=1e-6,
        )
    # 11 records, batch request 4 -> ceil(11/4)=3 draws/epoch -> q=1/3, floor(Nq)=3.
    np.testing.assert_allclose(baseline.sample_rate, 1 / 3)
    assert baseline.expected_batch_size == 3
    assert baseline.optimizer.param_groups[0]["lr"] == 0
    assert baseline.statistics["epsilon"] is None


def make_baseline(
    *, seed: int = 17, physical: int = 2, batch: int = 4
) -> MatchedBaseline:
    return MatchedBaseline(
        model_factory=public_model,
        optimizer_factory=lambda m: torch.optim.RMSprop(
            m.parameters(), lr=0.01, momentum=0.5
        ),
        inputs=torch.arange(1, 9, dtype=torch.float32).reshape(-1, 1) / 10,
        targets=torch.zeros(8, 1),
        recipe=BaselineRecipe(
            batch_size=batch,
            logical_steps=4,
            loss="mse",
            schedule="linear_decay",
            optimizer_definition="torch.RMSprop:lr0.01;momentum0.5",
        ),
        sampling_seed=seed,
        local_data_identity="public-eight-v1",
        max_physical_batch_size=physical,
    )


def test_empty_draw_still_updates_optimizer_and_schedule() -> None:
    baseline = make_baseline(seed=6, batch=1)
    baseline.trainer.train_local_steps(3)
    assert baseline.statistics["empty_draws"] == 3
    assert baseline.statistics["sampled_examples"] == 0
    assert baseline.trainer.completed_steps == 3
    np.testing.assert_allclose(baseline.optimizer.param_groups[0]["lr"], 0.0025)
    assert baseline.optimizer.state_dict()["state"][0]["step"] == 3
    np.testing.assert_array_equal(baseline.trainer.weights()["weight"], [[1]])
    assert baseline.trainer.local_loss == 0
    baseline.trainer.train_local_steps(1)
    assert baseline.statistics["sampled_examples"] == 1
    with pytest.raises(ValueError, match="horizon"):
        baseline.trainer.train_local_steps(1)


def test_group_optimizer_restore_and_physical_split_match() -> None:
    split = make_baseline()
    unsplit = make_baseline(physical=32)
    split.trainer.train_local_steps(2)
    saved = split.trainer.checkpoint_tensors()
    split.trainer.load_weights(
        split.trainer.weights()
    )  # Preserve RMSprop state across mixing.
    split.trainer.train_local_steps(2)
    unsplit.trainer.train_local_steps(4)
    restored = make_baseline()
    restored.trainer.load_checkpoint_tensors(saved)
    restored.trainer.train_local_steps(2)
    np.testing.assert_allclose(
        split.trainer.weights()["weight"],
        unsplit.trainer.weights()["weight"],
        atol=1e-7,
    )
    for name, value in split.trainer.checkpoint_tensors().items():
        np.testing.assert_array_equal(
            restored.trainer.checkpoint_tensors()[name], value
        )
    assert restored.statistics == split.statistics


def test_restore_rejects_disagreeing_counters_without_mutation() -> None:
    baseline = make_baseline()
    baseline.trainer.train_local_steps(1)
    old = baseline.trainer.checkpoint_tensors()
    bad = {k: v.copy() for k, v in old.items()}
    bad["meta.steps"][:] = 3
    with pytest.raises(ValueError, match="counters"):
        baseline.trainer.load_checkpoint_tensors(bad)
    for key, value in baseline.trainer.checkpoint_tensors().items():
        np.testing.assert_array_equal(value, old[key])


def digit_reference() -> DraftRunSpec:
    from support.application_fixture import application_draft

    from benchmarks.m3.matched_digits import TASK, model_definition, model_factory
    from dromeus.application import definition_hash
    from dromeus.manifests.models import (
        DraftRunSpec,
        PrivacyPolicy,
        PrivateOptimizerSpec,
        PrivateTrainingSpec,
    )
    from dromeus.training.private_trainer import private_training_definition

    values = application_draft().model_dump(mode="python")
    spec = PrivateTrainingSpec(
        optimizer=PrivateOptimizerSpec(
            name="adam",
            torch_version=torch.__version__,
            parameter_names=tuple(
                n for n, _ in model_factory("cpu", 17).named_parameters()
            ),
            learning_rate=0.001,
        ),
        loss="cross_entropy",
        batch_size=4,
        schedule="linear_decay",
        evaluation_provenance="public_fixed_holdout",
    )
    values.update(
        manifest_version=5,
        model_definition_hash=definition_hash(model_definition(17)),
        dataset={
            "dataset_id": "application-v1",
            "task_id": "real-digits",
            "definition_hash": definition_hash(TASK),
        },
        local_steps=1,
        round_count=2,
        privacy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=2,
            randomness_profile="reproducible_public_benchmark",
        ),
        private_training=spec,
        application_training={
            "policy_id": "owned-private-v1",
            "definition_hash": definition_hash(private_training_definition(spec)),
        },
    )
    values["environment"].update(
        model_definition_hash=values["model_definition_hash"],
        pytorch_version=torch.__version__,
    )
    values["algorithm_config"]["inner_steps"] = 1
    return DraftRunSpec.model_validate(values)


def test_digit_baseline_preserves_reference_and_exports_non_private_evidence(
    tmp_path: Path,
) -> None:
    import json

    from benchmarks.m3.matched_baseline import matched_draft
    from benchmarks.m3.matched_digits import SOURCES, execute, recipe_for

    reference = digit_reference()
    recipe = recipe_for(reference)
    draft = matched_draft(reference, recipe, run_id="matched-baseline-test")
    for field in (
        "model_definition_hash",
        "dataset",
        "environment",
        "local_steps",
        "round_count",
        "algorithm_config",
        "artifact_codecs",
        "peer_scheduler_seed",
        "consensus_sketch",
    ):
        assert getattr(draft, field) == getattr(reference, field)
    assert draft.privacy is None and draft.private_training is None
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(17)
    x = rng.random((8, 1, 32, 32), dtype=np.float32)
    y = np.arange(8, dtype=np.int64)
    np.savez(data / "mnist-train.npz", inputs=x, labels=y)
    for source in SOURCES:
        np.savez(data / f"{source}-test.npz", inputs=x[:2], labels=y[:2])
    root = tmp_path / "output"
    execute(
        mode="local",
        reference=reference,
        data=data,
        root=root,
        rank=0,
        device="cpu",
        seed=17,
        physical=2,
        run_id="matched-baseline-test",
        evaluation_interval=1,
    )
    result = json.loads((root / "baseline.json").read_text())
    assert result["logical_steps"] == 2 and result["epsilon"] is None
    assert result["sample_rate"] == 0.5 and result["expected_batch_size"] == 4
    assert len(result["history"]) == 3
    # Frozen public initialization from the accepted four-source pilot.
    assert (
        result["initial_checkpoint_sha256"]
        == "7e56ccdb9a5885671b0f47e33cb79d3d8dacd9ba88e5dceeb94db2d811347f56"
    )
    assert result["model_sha256"]
    assert result["source"] == "mnist"
    assert not list(root.glob("*ledger*"))


@pytest.mark.parametrize(
    "field,value",
    [("batch_size", 2), ("schedule", "constant"), ("optimizer_definition", "other")],
)
def test_baseline_rejects_mismatched_reference_recipe(
    field: str, value: object
) -> None:
    from benchmarks.m3.matched_baseline import matched_draft
    from benchmarks.m3.matched_digits import recipe_for

    reference = digit_reference()
    recipe = recipe_for(reference).model_copy(update={field: value})
    with pytest.raises(ValueError, match="differs"):
        matched_draft(reference, recipe, run_id="different-run")


def test_matched_group_optimizer_runs_through_four_noloco_nodes(tmp_path: Path) -> None:
    import asyncio

    from support.application_fixture import application_draft
    from support.in_memory_transport import InMemoryNetwork, InMemoryTransport

    from dromeus.application import definition_hash
    from dromeus.membership.formation import FormationResult, create_invitation
    from dromeus.runtime import (
        FailureConfig,
        InitiatorFormation,
        NodeRuntime,
        ParticipantFormation,
        TrainingConfig,
    )

    model_definition = "public-linear-weight1-no-bias-v1"
    task_definition = "public-eight-mse-v1"
    first = make_baseline()
    values = application_draft().model_dump(mode="python")
    values.update(
        manifest_version=5,
        local_steps=2,
        round_count=2,
        model_definition_hash=definition_hash(model_definition),
        dataset={
            "dataset_id": "application-v1",
            "task_id": "baseline-mse",
            "definition_hash": definition_hash(task_definition),
        },
        application_training={
            "policy_id": "matched-rmsprop",
            "definition_hash": definition_hash(first.recipe.training_definition),
        },
    )
    values["environment"].update(
        model_definition_hash=values["model_definition_hash"],
        pytorch_version=torch.__version__,
    )
    values["algorithm_config"]["inner_steps"] = 2
    draft = DraftRunSpec.model_validate(values)
    baselines = [first] + [make_baseline(seed=17 + i) for i in range(1, 4)]
    prepared = [
        b.prepare(
            draft,
            model_definition=model_definition,
            task_definition=task_definition,
            evaluation_interval=1,
        )
        for b in baselines
    ]
    checkpoint = prepared[0].create_initial_checkpoint(tmp_path / "initial.safetensors")
    network = InMemoryNetwork()
    nodes = [
        NodeRuntime(
            transport=InMemoryTransport(network=network, public_key=f"peer-{i}"),
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            artifact_root=tmp_path / f"formation-{i}",
            failure=FailureConfig.for_run_root(tmp_path / f"failure-{i}"),
        )
        for i in range(4)
    ]
    invitation = create_invitation(
        draft=draft, initiator_public_key="peer-0", bootstrap_uri="axl://memory"
    )

    async def run() -> None:
        async def node(rank: int) -> None:
            def training(result: FormationResult) -> TrainingConfig:
                return prepared[rank].build_config(
                    result=result,
                    local_public_key=f"peer-{rank}",
                    run_root=tmp_path / f"node-{rank}",
                )

            outcome = await nodes[rank].run_to_completion(
                formation=InitiatorFormation(
                    bootstrap_uri="axl://memory",
                    checkpoint_path=checkpoint.path,
                    tensor_schema=prepared[rank].tensor_schema,
                )
                if rank == 0
                else ParticipantFormation(invitation=invitation),
                training_factory=training,
            )
            assert len(outcome.commits) == 2
            assert baselines[rank].trainer.completed_steps == 4
            assert baselines[rank].statistics["epsilon"] is None
            assert baselines[rank].optimizer.state_dict()["state"][0]["step"] == 4

        await asyncio.wait_for(asyncio.gather(*(node(i) for i in range(4))), timeout=12)

    asyncio.run(run())


def test_evaluation_cadence_is_bound_and_mismatch_rejects() -> None:
    from benchmarks.m3.matched_baseline import matched_draft
    from benchmarks.m3.matched_digits import TASK, model_definition, recipe_for

    reference = digit_reference()
    slow = recipe_for(reference).model_copy(update={"evaluation_interval_rounds": 25})
    fast = recipe_for(reference).model_copy(update={"evaluation_interval_rounds": 1})
    assert slow.training_definition != fast.training_definition
    baseline = MatchedBaseline(
        model_factory=lambda: public_model(),
        optimizer_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
        inputs=torch.ones(2, 1),
        targets=torch.zeros(2, dtype=torch.int64),
        recipe=slow,
        sampling_seed=17,
        local_data_identity="cadence-test",
        max_physical_batch_size=1,
    )
    draft = matched_draft(reference, slow, run_id="cadence-run")
    with pytest.raises(ValueError, match="cadence"):
        baseline.prepare(
            draft,
            model_definition=model_definition(17),
            task_definition=TASK,
            evaluation_interval=1,
        )


def test_stochastic_model_rejects_before_benchmark_preparation() -> None:
    with pytest.raises(ValueError, match="deterministic"):
        MatchedBaseline(
            model_factory=lambda: nn.Sequential(nn.Dropout(0.5), nn.Linear(1, 1)),
            optimizer_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
            inputs=torch.ones(2, 1),
            targets=torch.zeros(2, 1),
            recipe=BaselineRecipe(
                batch_size=2,
                logical_steps=2,
                loss="mse",
                optimizer_definition="torch.SGD:lr0.1",
            ),
            sampling_seed=17,
            local_data_identity="stochastic-test",
            max_physical_batch_size=1,
        )
