"""Multi-task variant of :mod:`sleepwalker.models.MultiModel`.

``MetaModel`` reuses the embedding-fusion logic from ``MultiModel`` but replaces
the shared head with one task-specific head per configured task. Its main
consumer is ``MultiLabelTrainer`` in the standard multilabel configurations.
"""

import torch

from sleepwalker.models.MultiModel import MetaModelEntry, MultiModel


class MetaModel(MultiModel):
    """Fuse submodel embeddings and emit one categorical head per task.

    Args:
        task_config: Normalized task configuration whose entries provide at
            least ``labels`` and ``n_steps``.
        input_channels: Global input channel order expected by the composite
            model.
        models: Submodel entries that produce the fused embedding.
        preprocessors: Optional preprocessors applied before submodel slicing.

    Raises:
        ValueError: If a task config entry is missing the normalized fields
            required to build its output head.
    """

    def __init__(
        self,
        *,
        task_config: dict[str, dict],
        input_channels=None,
        models: list[MetaModelEntry],
        preprocessors=None,
    ):
        self.task_config = dict(task_config)
        classes = []
        for task, cfg in self.task_config.items():
            if "labels" not in cfg or "n_steps" not in cfg:
                raise ValueError(f"Task '{task}' must provide normalized config with 'labels' and 'n_steps'.")
            classes.extend(cfg["labels"])

        super().__init__(
            classes=classes,
            input_channels=input_channels,
            models=models,
            preprocessors=preprocessors,
        )
        del self.head
        self.heads = torch.nn.ModuleDict({
            task: torch.nn.Linear(self._feature_dim, cfg["n_steps"] * len(cfg["labels"]))
            for task, cfg in self.task_config.items()
        })

    def _classifier(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Project fused embeddings into task-specific logits.

        Returns:
            A mapping from task name to logits shaped
            ``[B, n_steps, n_labels]``.
        """
        out = {}
        for task, cfg in self.task_config.items():
            logits = self.heads[task](x)
            out[task] = logits.view(x.shape[0], cfg["n_steps"], len(cfg["labels"]))
        return out
