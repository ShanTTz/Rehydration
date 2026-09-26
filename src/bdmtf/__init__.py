"""BDMTF paper-aligned simulation package."""

from importlib import import_module
from typing import Any

__all__ = [
    "AgentProfile",
    "BDMTFSimulator",
    "Intervention",
    "RankingPolicy",
    "SimulationConfig",
]

_EXPORTS = {
    "AgentProfile": ("bdmtf.schema", "AgentProfile"),
    "Intervention": ("bdmtf.schema", "Intervention"),
    "RankingPolicy": ("bdmtf.schema", "RankingPolicy"),
    "SimulationConfig": ("bdmtf.schema", "SimulationConfig"),
    "BDMTFSimulator": ("bdmtf.simulator", "BDMTFSimulator"),
}


def __getattr__(name: str) -> Any:
    """Load the simulation stack only when its public API is requested."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
