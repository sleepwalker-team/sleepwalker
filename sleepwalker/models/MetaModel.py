import torch

from sleepwalker.models.MultiModel import MetaModelEntry, MultiModel


class MetaModel(MultiModel):
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
        out = {}
        for task, cfg in self.task_config.items():
            logits = self.heads[task](x)
            out[task] = logits.view(x.shape[0], cfg["n_steps"], len(cfg["labels"]))
        return out
