"""Closed registry of built-in training model recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import nn

from benchmarks.workloads.cifar10 import resnet18_groupnorm, resnet32


class ModelBuilder(Protocol):
    def __call__(
        self,
        *,
        seed: int,
        input_channels: int = 3,
        num_classes: int = 10,
    ) -> nn.Module: ...


@dataclass(frozen=True, slots=True)
class ModelRecipe:
    model_id: str
    definition: str
    definition_hash: str
    parameter_count: int
    tensor_schema_hash: str
    build: ModelBuilder


_RECIPES = {
    recipe.model_id: recipe
    for recipe in (
        ModelRecipe(
            model_id=resnet32.MODEL_ID,
            definition=resnet32.MODEL_DEFINITION,
            definition_hash=resnet32.MODEL_DEFINITION_HASH,
            parameter_count=resnet32.PARAMETER_COUNT,
            tensor_schema_hash=resnet32.TENSOR_SCHEMA_HASH,
            build=resnet32.build_model,
        ),
        ModelRecipe(
            model_id=resnet18_groupnorm.MODEL_ID,
            definition=resnet18_groupnorm.MODEL_DEFINITION,
            definition_hash=resnet18_groupnorm.MODEL_DEFINITION_HASH,
            parameter_count=resnet18_groupnorm.PARAMETER_COUNT,
            tensor_schema_hash=resnet18_groupnorm.TENSOR_SCHEMA_HASH,
            build=resnet18_groupnorm.build_model,
        ),
    )
}


def resolve_model(model_id: str, *, definition_hash: str | None = None) -> ModelRecipe:
    """Resolve and optionally verify one built-in model recipe."""
    try:
        recipe = _RECIPES[model_id]
    except KeyError as error:
        raise ValueError(f"unsupported training model: {model_id}") from error
    if definition_hash is not None and definition_hash != recipe.definition_hash:
        raise ValueError("model definition hash does not match built-in recipe")
    return recipe


__all__ = ["ModelRecipe", "resolve_model"]
