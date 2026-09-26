from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LocalModelConfig:
    model_platform: Any
    model_type: Any


class ModelFactory:
    @staticmethod
    def create(model_platform: Any, model_type: Any) -> LocalModelConfig:
        return LocalModelConfig(model_platform=model_platform, model_type=model_type)
