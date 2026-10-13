"""rtech-persona: SOUL/USER/KERNEL loader (scan + cap) and prompt slot composer.

Public API:
    load_persona - read + scan + cap SOUL/USER/KERNEL from an agent home.
    compose      - order persona + runtime slots into one system prompt.
    Persona      - the loaded-persona dataclass.
"""

from __future__ import annotations

from .composer import compose
from .config import CapConfig
from .loader import (
    MissingSoulError,
    Persona,
    PersonaError,
    PersonaScanError,
    load_persona,
)

__version__ = "0.0.1"

__all__ = [
    "CapConfig",
    "MissingSoulError",
    "Persona",
    "PersonaError",
    "PersonaScanError",
    "__version__",
    "compose",
    "load_persona",
]
