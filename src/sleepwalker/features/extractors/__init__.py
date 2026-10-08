"""The extractor plugin registry.

Drop a module in this folder that decorates its extractor class with ``@register_extractor("name")``
and it is **auto-discovered** on import -- no edits to the featurizer, the registry, or sibling
extractors. Modules whose name starts with ``_`` are treated as private helpers (e.g. a shared base
class) and are not scanned for registration.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Callable, Type

if TYPE_CHECKING:
    from sleepwalker.features.context import ExtractionContext
    from sleepwalker.features.extractor import Extractor

#: name -> Extractor subclass. Populated by ``@register_extractor`` as modules are discovered.
EXTRACTORS: dict[str, Type["Extractor"]] = {}


def register_extractor(name: str) -> Callable[[type], type]:
    """Class decorator registering an extractor under ``name``.

    Re-registering the *same* class is fine (idempotent on re-import); a *different* class under an
    existing name is an error.
    """

    def decorator(cls: type) -> type:
        existing = EXTRACTORS.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Extractor name {name!r} is already registered to {existing.__name__}."
            )
        EXTRACTORS[name] = cls
        return cls

    return decorator


def get_extractor(name: str, context: "ExtractionContext", **options) -> "Extractor":
    """Construct the registered extractor ``name`` with the shared context and its own options."""
    if name not in EXTRACTORS:
        raise ValueError(f"Unknown extractor {name!r}. Available: {sorted(EXTRACTORS)}.")
    return EXTRACTORS[name](context, **options)


def discover() -> None:
    """Import every public sibling module so its ``@register_extractor`` runs."""
    for _finder, modname, _ispkg in pkgutil.iter_modules(__path__):
        if not modname.startswith("_"):
            importlib.import_module(f"{__name__}.{modname}")


discover()
