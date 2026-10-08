"""The feature layer: one model-facing :class:`Featurizer` over pluggable :class:`Extractor`s.

``Featurizer(extractors, channel_roles, sample_frequency, ...)`` is the single config hub and plugin
host -- it builds and composes extractors by name. Add a new feature type by dropping a module in
``extractors/`` decorated with ``@register_extractor(name)``; it is auto-discovered and usable by name,
with no changes to the featurizer, the registry, or sibling extractors.
"""

from .context import ExtractionContext
from .extractor import Extractor
from .featurizer import Featurizer
from .extractors import EXTRACTORS, get_extractor, register_extractor

__all__ = [
    "Featurizer",
    "Extractor",
    "ExtractionContext",
    "EXTRACTORS",
    "get_extractor",
    "register_extractor",
]
