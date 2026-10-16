from __future__ import annotations

from ..core.config import AttackCategory
from .base import AttackStrategy
from .prompt_injection import PromptInjectionAttack

# Categories not listed here are declared in AttackCategory but not implemented yet.
STRATEGIES: dict[AttackCategory, type[AttackStrategy]] = {
    AttackCategory.PROMPT_INJECTION: PromptInjectionAttack,
}


def get_strategy(category: AttackCategory) -> AttackStrategy:
    try:
        return STRATEGIES[category]()
    except KeyError:
        implemented = ", ".join(c.value for c in STRATEGIES)
        raise NotImplementedError(
            f"Attack category '{category.value}' is not implemented yet (implemented: {implemented})"
        ) from None
