"""Predeclare fresh warning campaigns; calibrate and assess separate evidence.

This benchmark-only successor keeps the historical seed-17/29/41 loader intact.
It reuses its strict task/summary/scalar checks and warning-entry scoring, while
binding fresh seed sets and exact recipes through a separate predeclared spec.
No command starts training or authorizes cloud expenditure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.m3 import calibrate_detector as legacy

# Historical helpers are intentionally reused unchanged; source hashes pin this
# benchmark-only bridge until a separately reviewed shared-loader extraction.
from benchmarks.m3.calibrate_detector import (
    _assess as assess_entries,  # pyright: ignore[reportPrivateUsage]
)
from benchmarks.m3.calibrate_detector import (
    _bind_configuration as bind_configuration,  # pyright: ignore[reportPrivateUsage]
)
from benchmarks.m3.calibrate_detector import (
    _load_summary as load_summary,  # pyright: ignore[reportPrivateUsage]
)
from benchmarks.m3.calibrate_detector import (
    _observation as observation,  # pyright: ignore[reportPrivateUsage]
)
from benchmarks.m3.calibrate_detector import (
    _same as same_observation,  # pyright: ignore[reportPrivateUsage]
)
from benchmarks.m3.calibrate_detector import (
    _sources as sources,  # pyright: ignore[reportPrivateUsage]
)
from benchmarks.m3.final_report import DeclaredRun, load_run
from benchmarks.m3.final_runner import FrozenRunSpec
from benchmarks.m3.heterogeneous_data import SOURCES
from dromeus.manifests.canonical import (
    canonical_hash,
    file_sha256,
    validate_sealed_draft,
)
from dromeus.manifests.models import DivergencePolicy, SealedManifest
from dromeus.telemetry.consensus import ConsensusObservation
from dromeus.telemetry.evidence import (
    ConsensusObservationEvidence,
    DivergenceStatusEvidence,
    EvidenceError,
    EvidenceLog,
    NodeConsensusDistanceEvidence,
)

Split = Literal["calibration", "validation"]
ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "benchmarks/m3/detector_campaign.py",
    "benchmarks/m3/calibrate_detector.py",
    "benchmarks/m3/final_runner.py",
    "benchmarks/m3/final_report.py",
    "src/dromeus/manifests/canonical.py",
    "src/dromeus/manifests/models.py",
    "src/dromeus/telemetry/consensus.py",
    "src/dromeus/telemetry/divergence.py",
    "src/dromeus/telemetry/evidence.py",
)


class Criteria(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_delay_rounds: int = Field(default=10, ge=0)
    max_false_entries: Literal[0] = 0
    max_missed_fault_nodes: Literal[0] = 0
    grant_requirement: Literal[False] = False


class ExecutionEnvironment(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)
    nodes: Literal[4]
    torch: str = Field(min_length=1)
    axl_binary_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    gpu_profile: str = Field(min_length=1)
    grid_driver: str = Field(min_length=1)


class RuntimeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_hashes: dict[str, Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]] = Field(
        min_length=1
    )
    environment: ExecutionEnvironment


class RequiredRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    split: Split
    configuration: FrozenRunSpec
    participant_count: Literal[4] = 4
    participant_keys: tuple[str, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def public_recipe(self) -> Self:
        if len(set(self.participant_keys)) != self.participant_count:
            raise ValueError("required participant keys must be distinct")
        if self.configuration.detector is not None:
            raise ValueError("base recipe detector must be unset before selection")
        if self.configuration.steps % self.configuration.local_steps:
            raise ValueError("campaign requires a complete round horizon")
        return self


class CampaignSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1]
    campaign_id: str = Field(min_length=1)
    scoring_version: Literal["warning-entry-v1"]
    calibration_seeds: tuple[int, ...] = Field(min_length=1)
    validation_seeds: tuple[int, ...] = Field(min_length=1)
    runs: tuple[RequiredRun, ...] = Field(min_length=4)
    candidates: tuple[DivergencePolicy, ...] = Field(min_length=1, max_length=128)
    criteria: Criteria = Criteria()
    runtime_contract: RuntimeContract | None = None

    @model_validator(mode="after")
    def fresh_complete_contract(self) -> Self:
        calibration, validation = (
            set(self.calibration_seeds),
            set(self.validation_seeds),
        )
        if (
            len(calibration) != len(self.calibration_seeds)
            or len(validation) != len(self.validation_seeds)
            or calibration & validation
            or (calibration | validation) & {17, 29, 41}
        ):
            raise ValueError("split seed sets must be unique, disjoint and fresh")
        ids = [run.configuration.run_id for run in self.runs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate required run identity")
        for split, seeds in (("calibration", calibration), ("validation", validation)):
            selected = [r for r in self.runs if r.split == split]
            if {r.configuration.seed for r in selected} != seeds:
                raise ValueError("required runs must cover exactly the declared seeds")
            for seed in seeds:
                faults = [
                    r.configuration.fault is not None
                    for r in selected
                    if r.configuration.seed == seed
                ]
                if not faults or all(faults) or not any(faults):
                    raise ValueError("each seed needs healthy and faulted evidence")
        hashes = [canonical_hash(policy) for policy in self.candidates]
        if len(set(hashes)) != len(hashes):
            raise ValueError("candidate policies must be distinct")
        return self


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "freeze_hash": _digest(value)}


def _intact(value: dict[str, Any], status: str) -> None:
    if value.get("status") != status or value.get("freeze_hash") != _digest(
        {k: v for k, v in value.items() if k != "freeze_hash"}
    ):
        raise EvidenceError(f"requires an intact {status} freeze")


def freeze_spec(path: Path) -> dict[str, Any]:
    """Freeze recipes, finite candidates and scoring source before collection."""
    raw = path.read_bytes()
    spec = CampaignSpec.model_validate_json(raw)
    return _seal(
        {
            "version": 1,
            "status": "predeclared",
            "spec": spec.model_dump(mode="json"),
            "spec_file_sha256": hashlib.sha256(raw).hexdigest(),
            "source_hashes": {name: file_sha256(ROOT / name) for name in SOURCE_PATHS},
            "selection_order": [
                "false_entries",
                "misses",
                "late_detections",
                "max_delay",
                "mean_delay",
                "candidate_ordinal",
            ],
        }
    )


def _spec(predeclared: dict[str, Any]) -> CampaignSpec:
    _intact(predeclared, "predeclared")
    if predeclared.get("source_hashes") != {
        name: file_sha256(ROOT / name) for name in SOURCE_PATHS
    }:
        raise EvidenceError("campaign/scoring source differs from predeclared freeze")
    return CampaignSpec.model_validate(predeclared["spec"])


def _base(configuration: dict[str, Any]) -> dict[str, Any]:
    return {**configuration, "detector": None}


def _load(
    index_path: Path,
    spec: CampaignSpec,
    split: Split,
) -> tuple[legacy.LoadedRun, ...]:
    """Use legacy validation seams without rewriting seeds or historical files."""
    index_before = file_sha256(index_path)
    index = legacy.RunIndex.model_validate_json(index_path.read_bytes())
    required = {r.configuration.run_id: r for r in spec.runs if r.split == split}
    ids = [r.run_id for r in index.runs]
    if len(set(ids)) != len(ids) or set(ids) != set(required):
        raise EvidenceError("index must contain exactly the required distinct runs")
    seen_logs: set[str] = set()
    result: list[legacy.LoadedRun] = []
    for run in index.runs:
        expected = required[run.run_id]
        if (
            run.split != split
            or run.seed != expected.configuration.seed
            or _base(run.configuration)
            != expected.configuration.model_dump(mode="json")
        ):
            raise EvidenceError(
                "run split, seed or recipe differs from frozen campaign"
            )
        path = index_path.parent / run.manifest
        raw = path.read_bytes()
        manifest = SealedManifest.model_validate_json(raw)
        validate_sealed_draft(manifest)
        if (
            manifest.manifest_version != 5
            or manifest.run_id != run.run_id
            or manifest.peer_scheduler_seed != run.seed
            or manifest.round_count
            != expected.configuration.steps // expected.configuration.local_steps
            or manifest.participant_count != expected.participant_count
            or {p.public_key for p in manifest.participants}
            != set(expected.participant_keys)
        ):
            raise EvidenceError(
                "sealed manifest differs from campaign identity/horizon"
            )
        bind_configuration(run, manifest)
        if run.fault and run.fault.start_round >= manifest.round_count:
            raise EvidenceError("fault starts outside recorded horizon")
        participants = {p.public_key: p.node_index for p in manifest.participants}
        nodes: dict[str, legacy.NodeTrace] = {}
        distances: dict[tuple[str, int], NodeConsensusDistanceEvidence] = {}
        for ordinal, relative in enumerate(run.logs):
            log = index_path.parent / relative
            sha = file_sha256(log)
            if sha in seen_logs:
                raise EvidenceError("duplicate evidence log in campaign")
            seen_logs.add(sha)
            evidence = EvidenceLog.open(
                log, run_id=manifest.run_id, manifest_hash=canonical_hash(manifest)
            )
            if sha != file_sha256(log):
                raise EvidenceError("evidence changed while reading")
            node_id = evidence.node_id
            if node_id not in participants or node_id in nodes:
                raise EvidenceError("logs must identify distinct sealed participants")
            assert node_id is not None
            observations: dict[int, ConsensusObservation] = {}
            live: list[DivergenceStatusEvidence] = []
            for record in evidence.records:
                if isinstance(record, DivergenceStatusEvidence):
                    live.append(record)
                if not isinstance(
                    record,
                    (ConsensusObservationEvidence, NodeConsensusDistanceEvidence),
                ):
                    continue
                point = observation(record)
                if (
                    point.round_id >= manifest.round_count
                    or point.sketch_count != manifest.participant_count
                ):
                    raise EvidenceError(
                        "invalid observation round or participant count"
                    )
                previous = observations.get(point.round_id)
                if previous is not None and not same_observation(previous, point):
                    raise EvidenceError("scale-aware group/node measurements disagree")
                observations[point.round_id] = point
                if isinstance(record, NodeConsensusDistanceEvidence):
                    distances[node_id, point.round_id] = record
            if set(observations) != set(range(manifest.round_count)):
                raise EvidenceError("incomplete measurement coverage")
            nodes[node_id] = legacy.NodeTrace(
                node_id,
                tuple(observations[i] for i in range(manifest.round_count)),
                tuple(live),
                sha,
                str(log.resolve()),
                load_summary(
                    index_path.parent / run.summaries[ordinal], run, manifest, node_id
                )
                if run.summaries
                else None,
            )
        if set(nodes) != set(participants):
            raise EvidenceError("require one log for every sealed participant")
        ordered = tuple(sorted(nodes.values(), key=lambda n: participants[n.node_id]))
        if run.summaries and {
            n.summary.source_rank for n in ordered if n.summary
        } != set(range(len(SOURCES))):
            raise EvidenceError("terminal summaries must have distinct source ranks")
        for round_id in range(manifest.round_count):
            group = ordered[0].observations[round_id]
            if any(
                not same_observation(group, n.observations[round_id]) for n in ordered
            ):
                raise EvidenceError("nodes disagree on group scale measurements")
            individual = [
                distances[n.node_id, round_id]
                for n in ordered
                if (n.node_id, round_id) in distances
            ]
            if len(individual) == len(ordered) and not math.isclose(
                math.hypot(*(r.normalized_distance for r in individual))
                / math.sqrt(len(individual)),
                group.normalized_rms,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise EvidenceError(
                    "individual distances do not reconcile with group RMS"
                )
        result.append(
            legacy.LoadedRun(run, manifest, ordered, hashlib.sha256(raw).hexdigest())
        )
    if index_before != file_sha256(index_path):
        raise EvidenceError("index changed while loading")
    return tuple(result)


def _group_metrics(
    runs: tuple[legacy.LoadedRun, ...],
    details: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for run in runs:
        rows = [r for r in details if r["run_id"] == run.spec.run_id]
        groups.append(
            {
                "run_id": run.spec.run_id,
                "seed": run.spec.seed,
                "faulted": run.spec.fault is not None,
                "false_entries": sum(r["false_alerts"] for r in rows),
                "detected_on_all_nodes": run.spec.fault is not None
                and all(not r["missed_fault"] for r in rows),
            }
        )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "groups": len(rows),
            "healthy_groups": sum(not row["faulted"] for row in rows),
            "groups_with_false_entries": sum(row["false_entries"] > 0 for row in rows),
            "fault_groups": sum(row["faulted"] for row in rows),
            "fault_groups_detected_on_all_nodes": sum(
                row["detected_on_all_nodes"] for row in rows
            ),
        }

    return (
        {**aggregate(groups), "runs": groups, "node_repetitions_independent": False},
        {
            str(seed): aggregate([r for r in groups if r["seed"] == seed])
            for seed in sorted({r["seed"] for r in groups})
        },
    )


def _passed(metrics: dict[str, Any], criteria: Criteria) -> bool:
    return (
        metrics["false_alerts"] == 0
        and metrics["missed_fault_nodes"] == 0
        and metrics["fault_nodes"] > 0
        and metrics["max_delay_rounds"] is not None
        and metrics["max_delay_rounds"] <= criteria.max_delay_rounds
    )


def _occupancy(
    runs: tuple[legacy.LoadedRun, ...],
    details: list[dict[str, Any]],
) -> None:
    """Supplement entry scoring with complete-round active-warning occupancy."""
    horizons = {run.spec.run_id: run.manifest.round_count for run in runs}
    for node in details:
        transitions = iter(node["transitions"])
        next_transition = next(transitions, None)
        active = False
        active_rounds = 0
        for round_id in range(horizons[node["run_id"]]):
            while (
                next_transition is not None
                and next_transition["detection_round"] <= round_id
            ):
                active = next_transition["warning_active"]
                next_transition = next(transitions, None)
            active_rounds += int(active)
        node["warning_active_rounds"] = active_rounds
        node["warning_active_fraction"] = active_rounds / horizons[node["run_id"]]


def _same_decisions(
    replay: list[dict[str, Any]],
    live: list[dict[str, Any]],
) -> bool:
    """Compare source decisions, retaining async detection delay separately."""
    if len(replay) != len(live):
        return False
    for expected, actual in zip(replay, live, strict=True):
        for key, value in expected.items():
            if key in ("detection_round", "age_rounds"):
                continue
            measured = actual.get(key)
            if isinstance(value, float) and isinstance(measured, (float, int)):
                if not math.isclose(value, measured, rel_tol=1e-9, abs_tol=1e-12):
                    return False
            elif value != measured:
                return False
    return True


def calibrate(index: Path, predeclared: dict[str, Any]) -> dict[str, Any]:
    """Search only the declared calibration runs and finite candidate list."""
    spec = _spec(predeclared)
    runs = _load(index, spec, "calibration")
    scores: list[dict[str, Any]] = []
    for ordinal, policy in enumerate(spec.candidates):
        metrics, _ = assess_entries(runs, policy)
        score = [
            metrics["false_alerts"],
            metrics["missed_fault_nodes"],
            sum(d > spec.criteria.max_delay_rounds for d in metrics["delay_rounds"]),
            metrics["max_delay_rounds"] or 0,
            metrics["mean_delay_rounds"] or 0,
            ordinal,
        ]
        scores.append(
            {
                "candidate": ordinal,
                "score": score,
                "policy_hash": canonical_hash(policy),
            }
        )
    best = min(scores, key=lambda row: row["score"])
    chosen = spec.candidates[int(best["candidate"])]
    metrics, details = assess_entries(runs, chosen)
    _occupancy(runs, details)
    passed = _passed(metrics, spec.criteria)
    groups, seeds = _group_metrics(runs, details)
    return _seal(
        {
            "version": 1,
            "status": "frozen" if passed else "blocked",
            "campaign_freeze_hash": predeclared["freeze_hash"],
            "source_index_sha256": file_sha256(index),
            "policy": chosen.model_dump(mode="json") if passed else None,
            "policy_hash": canonical_hash(chosen) if passed else None,
            "best_candidate_policy": chosen.model_dump(mode="json"),
            "candidate_count": len(scores),
            "candidate_scores": scores,
            "metrics": metrics,
            "nodes": details,
            "group_metrics": groups,
            "seed_metrics": seeds,
            "sources": sources(runs),
            "engineering_criteria": spec.criteria.model_dump(mode="json"),
        }
    )


def evaluate(
    index: Path,
    predeclared: dict[str, Any],
    frozen: dict[str, Any],
    *,
    require_live: bool = False,
    runtime_root: Path | None = None,
) -> dict[str, Any]:
    """Assess fixed held-out evidence; replay success never certifies delivery."""
    spec = _spec(predeclared)
    _intact(frozen, "frozen")
    if frozen.get("campaign_freeze_hash") != predeclared["freeze_hash"]:
        raise EvidenceError("calibration freeze belongs to a different campaign")
    policy = DivergencePolicy.model_validate(frozen["policy"])
    if canonical_hash(policy) != frozen["policy_hash"] or policy not in spec.candidates:
        raise EvidenceError("frozen policy is not a predeclared candidate")
    runs = _load(index, spec, "validation")
    old_hashes = {n["sha256"] for r in frozen["sources"] for n in r["nodes"]}
    if any(n.sha256 in old_hashes for run in runs for n in run.nodes):
        raise EvidenceError("validation reuses calibration evidence")
    metrics, details = assess_entries(runs, policy)
    _occupancy(runs, details)
    groups, seeds = _group_metrics(runs, details)
    delivery = all(
        node.live
        and node.summary is not None
        and node.summary.event_delivery_dropped == 0
        and node.summary.detector_policy_hash == canonical_hash(policy)
        and run.manifest.divergence_detection == policy
        for run in runs
        for node in run.nodes
    )
    live_metrics: dict[str, Any] | None = None
    live_details: list[dict[str, Any]] | None = None
    replay_matches = False
    if delivery:
        for run in runs:
            for node in run.nodes:
                previous_round, previous_state = -1, "warming_up"
                for transition_id, record in enumerate(node.live, start=1):
                    if (
                        record.transition_id != transition_id
                        or record.previous_state != previous_state
                        or record.policy_hash != canonical_hash(policy)
                        or record.threshold_set_id != policy.threshold_set_id
                        or record.expected_participants
                        != run.manifest.participant_count
                        or not previous_round
                        <= record.detection_round
                        < run.manifest.round_count
                    ):
                        raise EvidenceError(
                            "live transition identity or ordering mismatch"
                        )
                    previous_round, previous_state = (
                        record.detection_round,
                        record.state,
                    )
        live_metrics, live_details = assess_entries(runs, policy, live=True)
        _occupancy(runs, live_details)
        replay_matches = all(
            _same_decisions(expected["transitions"], actual["transitions"])
            for expected, actual in zip(details, live_details, strict=True)
        )
        if not replay_matches:
            raise EvidenceError("live/replay source transitions do not reconcile")
    elif require_live:
        raise EvidenceError(
            "live assessment requires complete policy-bound zero-drop summaries"
        )
    live_warning_gate = bool(
        delivery
        and replay_matches
        and _passed(metrics, spec.criteria)
        and live_metrics is not None
        and _passed(live_metrics, spec.criteria)
    )
    runtime_proofs = (
        _runtime_proofs(runtime_root, spec, runs) if runtime_root is not None else None
    )
    if require_live and runtime_proofs is None:
        raise EvidenceError(
            "readiness requires independent runtime proof; supply --runtime-root "
            "and a predeclared runtime contract"
        )
    return {
        "version": 1,
        "campaign_freeze_hash": predeclared["freeze_hash"],
        "calibration_freeze_hash": frozen["freeze_hash"],
        "policy_hash": canonical_hash(policy),
        "source_index_sha256": file_sha256(index),
        "sources": sources(runs),
        "metrics": metrics,
        "nodes": details,
        "group_metrics": groups,
        "seed_metrics": seeds,
        "live_metrics": live_metrics,
        "live_nodes": live_details,
        "live_delivery_verified": bool(delivery),
        "live_replay_matches": replay_matches,
        "replay_gate_passed": _passed(metrics, spec.criteria),
        "live_warning_gate_passed": live_warning_gate,
        "runtime_proof_verified": runtime_proofs is not None,
        "runtime_proofs": runtime_proofs,
        "readiness_passed": live_warning_gate and runtime_proofs is not None,
        "limitations": [
            "No-injection labels do not prove useful learning.",
            "Node monitors and repeated profiles sharing a seed are correlated.",
            "A pre-active warning is not a new fault detection.",
            "Small workload-specific suite; no population false-alert guarantee.",
            "Shared model collapse can remain invisible to disagreement.",
            "Live statuses alone do not prove completed training, fault execution "
            "or deployment provenance; full runtime proof is independently required.",
        ],
    }


def _runtime_proofs(
    root: Path,
    spec: CampaignSpec,
    runs: tuple[legacy.LoadedRun, ...],
) -> list[dict[str, Any]]:
    """Reuse full runtime audit, then bind its artifacts to the assessed logs.

    The retained layout is root/runs/<run-id>/rank-<source-rank>/. load_run
    checks terminal committed horizon, actual fault injections, model/input and
    accounting identities, four distinct GPU workers, frozen deployment sources,
    AXL/Torch/driver versions and final model-versus-sketch reconciliation.
    """
    contract = spec.runtime_contract
    if contract is None:
        raise EvidenceError("runtime proof requires a predeclared runtime contract")
    proofs: list[dict[str, Any]] = []
    for run in runs:
        configuration = FrozenRunSpec.model_validate_json(
            json.dumps(run.spec.configuration)
        )
        declared = DeclaredRun(
            role="validation_fault" if run.spec.fault else "validation_healthy",
            config=configuration,
        )
        try:
            audit = load_run(
                root,
                declared,
                contract.source_hashes,
                contract.environment.model_dump(mode="json"),
            )
        except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
            raise EvidenceError(
                f"runtime proof failed for {run.spec.run_id}: {error}"
            ) from error
        if (
            audit["manifest_hash"] != canonical_hash(run.manifest)
            or audit["config_hash"] != canonical_hash(configuration)
            or audit["run_id"] != run.spec.run_id
        ):
            raise EvidenceError("runtime proof binds a different run or manifest")
        by_id = {node["node_id"]: node for node in audit["nodes"]}
        if len(by_id) != len(run.nodes) or set(by_id) != {n.node_id for n in run.nodes}:
            raise EvidenceError("runtime proof must cover every assessed participant")
        nodes: list[dict[str, Any]] = []
        for node in run.nodes:
            proof = by_id[node.node_id]
            hashes = proof["source_hashes"]
            if (
                node.summary is None
                or proof["rank"] != node.summary.source_rank
                or proof["event_delivery_dropped"] != 0
                or hashes["logs/dromeus.jsonl"] != node.sha256
                or hashes["summary.json"] != node.summary.sha256
                or hashes["run-store/manifest.json"] != run.manifest_sha256
            ):
                raise EvidenceError(
                    "runtime proof artifacts differ from assessed evidence"
                )
            nodes.append(
                {
                    "node_id": node.node_id,
                    "rank": proof["rank"],
                    "source_hashes": hashes,
                    "runtime_provenance": proof["runtime_provenance"],
                }
            )
        proofs.append(
            {
                "run_id": run.spec.run_id,
                "manifest_hash": audit["manifest_hash"],
                "config_hash": audit["config_hash"],
                "nodes": nodes,
                "final_consensus": audit["final_consensus"],
            }
        )
    return proofs


def template() -> dict[str, Any]:
    """Describe required real metadata without inventing a launchable freeze."""
    return {
        "status": "template_requires_measured_metadata",
        "instructions": (
            "Create a CampaignSpec JSON with actual FrozenRunSpec recipes, peer keys, "
            "fresh disjoint seeds and measured finite DivergencePolicy candidates; "
            "run freeze-spec before collecting calibration evidence. This is not "
            "authorization to train or spend. Readiness additionally requires a "
            "predeclared runtime_contract and the full retained runtime evidence tree."
        ),
        "schema": CampaignSpec.model_json_schema(),
        "commands": [
            "python -m benchmarks.m3.detector_campaign freeze-spec "
            "--spec campaign.json --output predeclared.json",
            "python -m benchmarks.m3.detector_campaign calibrate "
            "--predeclared predeclared.json --index calibration-index.json "
            "--output frozen.json",
            "python -m benchmarks.m3.detector_campaign evaluate "
            "--predeclared predeclared.json --frozen frozen.json "
            "--index validation-index.json --require-live --runtime-root evidence "
            "--output assessment.json",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("template", "freeze-spec", "calibrate", "evaluate")
    )
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--predeclared", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "template":
        report = template()
    elif args.action == "freeze-spec":
        if args.spec is None:
            parser.error("freeze-spec requires --spec")
        report = freeze_spec(args.spec)
    else:
        if args.predeclared is None or args.index is None:
            parser.error("calibrate/evaluate require --predeclared and --index")
        predeclared = json.loads(args.predeclared.read_bytes())
        if args.action == "calibrate":
            report = calibrate(args.index, predeclared)
        else:
            if args.frozen is None:
                parser.error("evaluate requires --frozen")
            report = evaluate(
                args.index,
                predeclared,
                json.loads(args.frozen.read_bytes()),
                require_live=args.require_live,
                runtime_root=args.runtime_root,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if (
        report.get("status") == "blocked"
        or (args.action == "evaluate" and not report["replay_gate_passed"])
        or (
            args.action == "evaluate"
            and args.require_live
            and not report["readiness_passed"]
        )
    ):
        raise SystemExit("campaign gate failed; evidence retained in output")


if __name__ == "__main__":
    main()
