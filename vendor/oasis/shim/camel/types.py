from __future__ import annotations

from enum import Enum


class ModelPlatformType(str, Enum):
    OPENAI = "OPENAI"


class ModelType(str, Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
