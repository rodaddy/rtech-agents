"""Typed, immutable configuration for persona loading and composition.

Values port the intent of Hermes's context-file cap behavior
(``prompt_builder._dynamic_context_file_max_chars`` /
``_truncate_content``): a floor, a context-window-scaled budget, and a ceiling,
with a head/tail keep ratio when truncation is required.
"""

from __future__ import annotations

from dataclasses import dataclass

# Persona source filenames (relative to the agent home).
SOUL_FILENAME = "SOUL.md"
USER_FILENAME = "USER.md"
KERNEL_FILENAME = "KERNEL.md"


@dataclass(frozen=True)
class CapConfig:
    """Length-cap parameters for a single persona file.

    The cap scales with the model's context budget so large-context models
    rarely truncate, while small-context models stay at the historical floor.
    ``chars_per_token`` is the usual ~4 English heuristic; ``window_fraction``
    is the slice of the window spent on a persona file.
    """

    floor_chars: int = 20_000
    ceiling_chars: int = 500_000
    chars_per_token: int = 4
    window_fraction: float = 0.06
    truncate_head_ratio: float = 0.7
    truncate_tail_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.floor_chars <= 0:
            raise ValueError(f"floor_chars must be positive, got {self.floor_chars!r}")
        if self.ceiling_chars < self.floor_chars:
            raise ValueError(
                f"ceiling_chars ({self.ceiling_chars}) must be >= floor_chars ({self.floor_chars})"
            )
        if self.chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be positive, got {self.chars_per_token!r}")
        if not 0.0 < self.window_fraction <= 1.0:
            raise ValueError(f"window_fraction must be in (0, 1], got {self.window_fraction!r}")
        total_ratio = self.truncate_head_ratio + self.truncate_tail_ratio
        if not 0.0 < total_ratio <= 1.0:
            raise ValueError(f"truncate head+tail ratio must be in (0, 1], got {total_ratio!r}")

    def max_chars(self, context_budget: int | None) -> int:
        """Return the char cap for the given context budget (tokens).

        Args:
            context_budget: The model's context window in tokens, or None/<=0
                to use the flat floor (behavior unchanged from the default).

        Returns:
            A char count between ``floor_chars`` and ``ceiling_chars``.
        """
        if not isinstance(context_budget, int) or context_budget <= 0:
            return self.floor_chars
        budget = int(context_budget * self.chars_per_token * self.window_fraction)
        return max(self.floor_chars, min(budget, self.ceiling_chars))


DEFAULT_CAP = CapConfig()
