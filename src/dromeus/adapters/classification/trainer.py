"""Training from independently held, already preprocessed classification data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from threading import RLock
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from dromeus.adapters.classification.data import (
    LocalClassificationData,
    ValidatedLocalData,
    validate_local_data,
)
from dromeus.adapters.classification.torch_trainer import (
    InitialCheckpoint,
    PyTorchTrainer,
    TrainerSettings,
    create_initial_checkpoint,
)
from dromeus.manifests.canonical import canonical_hash, file_sha256, load_safetensors
from dromeus.manifests.models import (
    ClassificationTaskContract,
    DraftRunSpec,
    SealedManifest,
    TensorSchema,
)
from dromeus.training.model_state import tensor_schema_for_model

_LOCAL_STATE = "__dromeus_local__."
_FINGERPRINT = f"{_LOCAL_STATE}data_fingerprint"
_CONTEXT = f"{_LOCAL_STATE}restore_context_v1"
_OPTIMIZER_KEYS = f"{_LOCAL_STATE}optimizer_keys_v1"
_CPU_RNG = f"{_LOCAL_STATE}torch_cpu_rng"
_CUDA_RNG = f"{_LOCAL_STATE}torch_cuda_rng"
_TRAINING_STATE = "__dromeus_training__."
_COMPLETED_STEPS = f"{_TRAINING_STATE}completed_steps"
_BATCHES_CONSUMED = f"{_TRAINING_STATE}batches_consumed"
_AUGMENTATION_RNG = f"{_TRAINING_STATE}augmentation_rng"
_LOADER_EPOCH_RNG = f"{_TRAINING_STATE}loader_epoch_rng"
_TRAINER_METADATA = {
    _COMPLETED_STEPS,
    _BATCHES_CONSUMED,
    _AUGMENTATION_RNG,
    _LOADER_EPOCH_RNG,
}
_TORCH_RNG_LOCK = RLock()
_fork_rng = cast(
    Callable[..., AbstractContextManager[None]],
    torch.random.fork_rng,  # pyright: ignore[reportUnknownMemberType]
)


def _local_task(draft: DraftRunSpec) -> ClassificationTaskContract:
    if draft.manifest_version != 4 or not isinstance(
        draft.dataset, ClassificationTaskContract
    ):
        raise ValueError("local training requires a manifest v4 classification task")
    policy = draft.training
    if policy is None or policy.crop_padding != 0 or policy.normalize:
        raise ValueError(
            "local inputs are preprocessed: crop_padding must be 0 and normalize false"
        )
    return draft.dataset


def _trainer_settings(
    manifest: SealedManifest, *, seed: int, device: str
) -> TrainerSettings:
    _local_task(manifest)
    if manifest.optimizer == "application":
        raise ValueError("classification adapter requires its own training policy")
    policy = manifest.training
    assert policy is not None
    config = manifest.algorithm_config
    adam = config.adam if manifest.optimizer == "adam" and config is not None else None
    return TrainerSettings(
        seed=seed,
        batch_size=policy.batch_size,
        learning_rate=adam.learning_rate
        if adam is not None
        else manifest.require_learning_rate(),
        optimizer=manifest.optimizer,
        momentum=policy.momentum if manifest.optimizer == "sgd" else 0.0,
        weight_decay=policy.weight_decay,
        adam_beta1=adam.beta1 if adam is not None else 0.9,
        adam_beta2=adam.beta2 if adam is not None else 0.999,
        adam_epsilon=adam.epsilon if adam is not None else 1e-8,
        gradient_clip_norm=adam.gradient_clip_norm if adam is not None else None,
        learning_rate_milestones=policy.learning_rate_milestones,
        learning_rate_gamma=policy.learning_rate_gamma,
        learning_rate_schedule=policy.learning_rate_schedule,
        device=device,
        augment=False,
    )


def _cuda_device(device: str) -> int | None:
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("local training supports CPU or CUDA devices")
    if target.type == "cpu":
        return None
    if not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    index = cast(int | None, target.index)
    return index if index is not None else torch.cuda.current_device()


def _restore_context(
    manifest: SealedManifest, settings: TrainerSettings, model: nn.Module
) -> str:
    """Bind run-independent semantics, including exact seed/device policy."""
    values = {field.name: getattr(settings, field.name) for field in fields(settings)}
    values["learning_rate_schedule"] = (
        None
        if settings.learning_rate_schedule is None
        else settings.learning_rate_schedule.model_dump(mode="json")
    )
    payload = {
        "version": 1,
        "model_id": manifest.model_id,
        "model_definition_hash": manifest.model_definition_hash,
        "tensor_schema": manifest.tensor_schema.model_dump(mode="json"),
        "parameters": [
            {"name": name, "requires_grad": parameter.requires_grad}
            for name, parameter in model.named_parameters()
        ],
        "task": manifest.dataset.model_dump(mode="json"),
        "training_policy": (
            None
            if manifest.training is None
            else manifest.training.model_dump(mode="json")
        ),
        "trainer_settings": values,
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _validate_model(
    model: nn.Module, *, sample: Tensor, class_count: int, device: str
) -> TensorSchema:
    state = model.state_dict()
    if not state or not any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("local model requires trainable FP32 state")
    for name, value in state.items():
        if name.startswith((_LOCAL_STATE, "__dromeus_training__.")):
            raise ValueError("model state name uses a reserved checkpoint namespace")
        if value.ndim == 0:
            raise ValueError(
                "local model state must be non-scalar; use shape [1] for scalar values"
            )
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError("local model state must contain only finite FP32 tensors")
    tensor_schema = tensor_schema_for_model(model)
    probe = deepcopy(model)
    modes = [(module, module.training) for module in probe.modules()]
    cuda_device = _cuda_device(device)
    try:
        probe.eval()
        with (
            _TORCH_RNG_LOCK,
            _fork_rng(devices=[] if cuda_device is None else [cuda_device]),
            torch.no_grad(),
        ):
            try:
                output = probe(sample.unsqueeze(0).to(device))
            except (RuntimeError, TypeError, ValueError) as error:
                raise ValueError(
                    "local model cannot evaluate the declared input schema"
                ) from error
        if (
            not isinstance(output, Tensor)
            or tuple(output.shape) != (1, class_count)
            or output.dtype != torch.float32
            or not bool(torch.isfinite(output).all())
        ):
            raise ValueError(
                "local classifier must return finite FP32 logits "
                "shaped [1, class_count]"
            )
    finally:
        for module, training in modes:
            module.training = training
    return tensor_schema


@dataclass(frozen=True, slots=True)
class PreparedLocalTraining:
    """Private validated workload bound to one shared draft identity."""

    draft_hash: str
    tensor_schema: TensorSchema
    data_metadata: ValidatedLocalData
    _data: LocalClassificationData
    _task: ClassificationTaskContract
    _model: nn.Module
    _model_definition: str
    _seed: int
    _device: str

    def validate_draft(self, draft: DraftRunSpec) -> None:
        if canonical_hash(draft) != self.draft_hash:
            raise ValueError("draft does not match prepared local training")

    def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
        return create_initial_checkpoint(
            path, model=self._model, model_definition=self._model_definition
        )

    def create_trainer(
        self, *, manifest: SealedManifest, local_public_key: str
    ) -> LocalClassificationTrainer:
        if _local_task(manifest) != self._task:
            raise ValueError("sealed task does not match prepared local training")
        if manifest.draft_hash != self.draft_hash:
            raise ValueError("sealed draft hash does not match prepared local training")
        # Verify the actual sealed fields too, rather than trusting its hash field.
        formed_draft = DraftRunSpec.model_validate(
            {name: getattr(manifest, name) for name in DraftRunSpec.model_fields}
        )
        self.validate_draft(formed_draft)
        node_index = next(
            (
                member.node_index
                for member in manifest.participants
                if member.public_key == local_public_key
            ),
            None,
        )
        if node_index is None:
            raise ValueError("local public key is not a sealed participant")
        if manifest.tensor_schema != self.tensor_schema:
            raise ValueError("sealed tensor schema does not match prepared local model")
        current_data = validate_local_data(self._data, self._task)
        if current_data.fingerprint != self.data_metadata.fingerprint:
            raise ValueError("local dataset changed after preparation")
        settings = _trainer_settings(
            manifest, seed=self._seed + node_index, device=self._device
        )
        template = deepcopy(self._model)

        def create_backend() -> PyTorchTrainer:
            return PyTorchTrainer(
                model=deepcopy(template),
                model_definition=self._model_definition,
                train_data=current_data.train_data,
                test_data=current_data.evaluation_data,
                settings=settings,
            )

        return LocalClassificationTrainer(
            backend_factory=create_backend,
            data=self._data,
            task=self._task,
            metadata=current_data,
            initial_checkpoint_hash=manifest.initial_checkpoint_hash,
            restore_context=_restore_context(manifest, settings, template),
            parameter_shapes={
                name: tuple(parameter.shape)
                for name, parameter in template.named_parameters()
                if parameter.requires_grad
            },
        )


def prepare_local_training(
    *,
    draft: DraftRunSpec,
    data: LocalClassificationData,
    model: nn.Module,
    model_definition: str,
    seed: int,
    device: str = "cpu",
) -> PreparedLocalTraining:
    """Validate local records and a custom classifier before formation can start."""
    task = _local_task(draft)
    definition_hash = hashlib.sha256(model_definition.encode()).hexdigest()
    if definition_hash != draft.model_definition_hash:
        raise ValueError("local model definition hash does not match draft")
    metadata = validate_local_data(data, task)
    _cuda_device(device)
    prepared_model = deepcopy(model).to(device)
    tensor_schema = _validate_model(
        prepared_model,
        sample=metadata.train_data[0][0],
        class_count=task.class_count,
        device=device,
    )
    return PreparedLocalTraining(
        draft_hash=canonical_hash(draft),
        tensor_schema=tensor_schema,
        data_metadata=metadata,
        _data=data,
        _task=task,
        _model=prepared_model,
        _model_definition=model_definition,
        _seed=seed,
        _device=device,
    )


def _rng_tensor(value: np.ndarray, *, device: str) -> Tensor:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ValueError("local model RNG state is invalid")
    tensor = torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
        cast(Any, np.ascontiguousarray(array).copy())
    )
    try:
        torch.Generator(device=device).set_state(tensor)
    except RuntimeError as error:
        raise ValueError("local model RNG state is invalid") from error
    return tensor


class LocalClassificationTrainer:
    """Generic trainer with explicit held-out evaluation and private data binding."""

    def __init__(
        self,
        *,
        backend_factory: Callable[[], PyTorchTrainer],
        data: LocalClassificationData,
        task: ClassificationTaskContract,
        metadata: ValidatedLocalData,
        initial_checkpoint_hash: str,
        restore_context: str,
        parameter_shapes: dict[str, tuple[int, ...]],
    ) -> None:
        self._backend_factory = backend_factory
        self._trainer = trainer = backend_factory()
        self._data = data
        self._task = task
        self._metadata = metadata
        self._initial_checkpoint_hash = initial_checkpoint_hash
        self._restore_context = restore_context
        self._parameter_shapes = dict(parameter_shapes)
        self._cpu_rng = (
            torch.Generator().manual_seed(trainer.settings.seed + 3).get_state()
        )
        self._cuda_device = _cuda_device(trainer.settings.device)
        self._cuda_rng = (
            None
            if self._cuda_device is None
            else torch.Generator(device=f"cuda:{self._cuda_device}")
            .manual_seed(trainer.settings.seed + 4)
            .get_state()
        )

    @property
    def tensor_schema(self) -> TensorSchema:
        return self._trainer.tensor_schema

    @property
    def settings(self) -> TrainerSettings:
        return self._trainer.settings

    @property
    def local_loss(self) -> float | None:
        return self._trainer.local_loss

    @property
    def learning_rate(self) -> float:
        return self._trainer.learning_rate

    def weights(self) -> dict[str, np.ndarray]:
        return self._trainer.weights()

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._validate_weights(weights)
        self._trainer.load_weights(weights)

    def _validate_weights(self, weights: dict[str, np.ndarray]) -> None:
        schema = self.tensor_schema.tensors
        if set(weights) != {tensor.name for tensor in schema}:
            raise ValueError("local weight names do not match model")
        for tensor in schema:
            array = np.asarray(weights[tensor.name])
            if (
                array.dtype != np.float32
                or array.shape != tensor.shape
                or not np.isfinite(array).all()
            ):
                raise ValueError(
                    "local weights must match the finite FP32 model schema"
                )

    def train_local_steps(self, step_count: int) -> None:
        with self._model_rng(advance=True):
            self._trainer.train_local_steps(step_count)

    def evaluate(self) -> tuple[float, float] | None:
        if self._metadata.evaluation_data is None:
            return None
        with self._model_rng(advance=False):
            return self._trainer.evaluate(self._metadata.evaluation_data)

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        # Match the durable store's scalar normalization (notably Adam steps).
        state = {
            key: np.ascontiguousarray(value)
            for key, value in self._trainer.checkpoint_tensors().items()
        }
        state[_FINGERPRINT] = np.frombuffer(
            bytes.fromhex(self._metadata.fingerprint), dtype=np.uint8
        ).copy()
        state[_CONTEXT] = np.frombuffer(
            bytes.fromhex(self._restore_context), dtype=np.uint8
        ).copy()
        optimizer_keys = sorted(
            key
            for key in state
            if key.startswith(_TRAINING_STATE) and key not in _TRAINER_METADATA
        )
        state[_OPTIMIZER_KEYS] = np.frombuffer(
            json.dumps(optimizer_keys, separators=(",", ":")).encode(), dtype=np.uint8
        ).copy()
        state[_CPU_RNG] = self._cpu_rng.numpy().copy()
        if self._cuda_rng is not None:
            state[_CUDA_RNG] = self._cuda_rng.numpy().copy()
        return state

    def _validate_current_data(self) -> None:
        current = validate_local_data(self._data, self._task)
        if current.fingerprint != self._metadata.fingerprint:
            raise ValueError("local dataset changed since trainer construction")

    def load_checkpoint_tensors(self, state: dict[str, np.ndarray]) -> None:
        # Check both current records and checkpoint identity before model mutation.
        self._validate_current_data()
        state = {key: value.copy() for key, value in state.items()}
        fingerprint = state.get(_FINGERPRINT)
        if (
            fingerprint is None
            or fingerprint.dtype != np.uint8
            or fingerprint.shape != (32,)
            or fingerprint.tobytes().hex() != self._metadata.fingerprint
        ):
            raise ValueError(
                "checkpoint local data fingerprint is missing or mismatched"
            )
        context = state.get(_CONTEXT)
        if (
            context is None
            or context.dtype != np.uint8
            or context.shape != (32,)
            or context.tobytes().hex() != self._restore_context
        ):
            raise ValueError(
                "checkpoint model or trainer restore context does not match"
            )
        expected = {_FINGERPRINT, _CONTEXT, _OPTIMIZER_KEYS, _CPU_RNG}
        if self._cuda_device is not None:
            expected.add(_CUDA_RNG)
        if {key for key in state if key.startswith(_LOCAL_STATE)} != expected:
            raise ValueError("local checkpoint metadata is incomplete or incompatible")
        cpu_rng = _rng_tensor(state[_CPU_RNG], device="cpu")
        cuda_rng = (
            None
            if self._cuda_device is None
            else _rng_tensor(state[_CUDA_RNG], device=f"cuda:{self._cuda_device}")
        )
        self._validate_trainer_state(state, expected)
        devices = [] if self._cuda_device is None else [self._cuda_device]
        with _TORCH_RNG_LOCK, _fork_rng(devices=devices):
            candidate = self._backend_factory()
            try:
                candidate.load_checkpoint_tensors(
                    {key: value for key, value in state.items() if key not in expected}
                )
            except (RuntimeError, TypeError, IndexError, OverflowError) as error:
                raise ValueError(
                    "local trainer checkpoint cannot be restored"
                ) from error
        self._trainer = candidate
        self._cpu_rng = cpu_rng
        self._cuda_rng = cuda_rng

    def _validate_trainer_state(
        self, state: dict[str, np.ndarray], local_keys: set[str]
    ) -> None:
        model_names = {tensor.name for tensor in self.tensor_schema.tensors}
        if not model_names.issubset(state) or not _TRAINER_METADATA.issubset(state):
            raise ValueError("local trainer checkpoint is incomplete")
        self._validate_weights({name: state[name] for name in model_names})
        counters: dict[str, int] = {}
        for name in (_COMPLETED_STEPS, _BATCHES_CONSUMED):
            value = state[name]
            if value.dtype != np.int64 or value.shape != (1,) or int(value[0]) < 0:
                raise ValueError("local trainer checkpoint counter is invalid")
            counters[name] = int(value[0])
        steps = counters[_COMPLETED_STEPS]
        batches = (
            self._metadata.train_sample_count + self.settings.batch_size - 1
        ) // self.settings.batch_size
        consumed = 0 if steps == 0 else (steps - 1) % batches + 1
        if counters[_BATCHES_CONSUMED] != consumed:
            raise ValueError("local checkpoint loader position disagrees with steps")
        schedule = self.settings.learning_rate_schedule
        if schedule is not None and steps > schedule.total_inner_steps:
            raise ValueError("local checkpoint exceeds the training schedule")
        for name in (_AUGMENTATION_RNG, _LOADER_EPOCH_RNG):
            _rng_tensor(state[name], device="cpu")
        keys_value = state[_OPTIMIZER_KEYS]
        if keys_value.dtype != np.uint8 or keys_value.ndim != 1:
            raise ValueError("local checkpoint optimizer inventory is invalid")
        try:
            raw = cast(object, json.loads(keys_value.tobytes()))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                "local checkpoint optimizer inventory is invalid"
            ) from error
        if not isinstance(raw, list):
            raise ValueError("local checkpoint optimizer inventory is invalid")
        items = cast(list[object], raw)
        if any(not isinstance(item, str) for item in items):
            raise ValueError("local checkpoint optimizer inventory is invalid")
        optimizer_keys = cast(list[str], items)
        if optimizer_keys != sorted(set(optimizer_keys)):
            raise ValueError("local checkpoint optimizer inventory is invalid")
        if set(state) != model_names | _TRAINER_METADATA | local_keys | set(
            optimizer_keys
        ):
            raise ValueError(
                "local checkpoint tensors disagree with optimizer inventory"
            )
        allowed: set[str] = set()
        for name, shape in self._parameter_shapes.items():
            if self.settings.optimizer == "sgd":
                names = {f"{_TRAINING_STATE}momentum.{name}"}
                if self.settings.momentum == 0.0:
                    continue
            else:
                names = {
                    f"{_TRAINING_STATE}adam.{part}.{name}"
                    for part in ("exp_avg", "exp_avg_sq", "step")
                }
            allowed.update(names)
            present = names & set(optimizer_keys)
            if not present:
                continue
            if present != names or steps == 0:
                raise ValueError("local checkpoint optimizer state is incomplete")
            for key in names:
                value = state[key]
                is_step = key.startswith(f"{_TRAINING_STATE}adam.step.")
                expected_shape = (1,) if is_step else shape
                if (
                    value.dtype != np.float32
                    or value.shape != expected_shape
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("local checkpoint optimizer tensor is invalid")
                if is_step:
                    step = float(value[0])
                    if not step.is_integer() or not 1 <= step <= steps:
                        raise ValueError("local checkpoint Adam step is invalid")
                elif key.startswith(f"{_TRAINING_STATE}adam.exp_avg_sq."):
                    if np.any(value < 0):
                        raise ValueError("local checkpoint Adam variance is negative")
        if set(optimizer_keys) - allowed:
            raise ValueError("local checkpoint has unsupported optimizer tensors")

    def load_checkpoint(self, path: Path) -> None:
        """Load the sealed shared initialization; durable restore uses tensor state."""
        self._validate_current_data()
        if file_sha256(path) != self._initial_checkpoint_hash:
            raise ValueError("initial checkpoint hash does not match sealed manifest")
        weights = load_safetensors(str(path))
        self._validate_weights(weights)
        devices = [] if self._cuda_device is None else [self._cuda_device]
        with _TORCH_RNG_LOCK, _fork_rng(devices=devices):
            candidate = self._backend_factory()
            candidate.load_weights(weights)
        self._trainer = candidate

    @contextmanager
    def _model_rng(self, *, advance: bool) -> Generator[None, None, None]:
        devices = [] if self._cuda_device is None else [self._cuda_device]
        with _TORCH_RNG_LOCK, _fork_rng(devices=devices):
            torch.set_rng_state(self._cpu_rng)
            if self._cuda_rng is not None and self._cuda_device is not None:
                torch.cuda.set_rng_state(self._cuda_rng, self._cuda_device)
            try:
                yield
            finally:
                if advance:
                    self._cpu_rng = torch.get_rng_state().clone()
                    if self._cuda_device is not None:
                        self._cuda_rng = torch.cuda.get_rng_state(
                            self._cuda_device
                        ).clone()


__all__ = [
    "LocalClassificationTrainer",
    "PreparedLocalTraining",
    "prepare_local_training",
]
