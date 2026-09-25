"""One shared embedding model with one classification head per task."""

import torch

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel


class MultiTaskClassifierModel(BaseModel, ClassifierModel):
    def __init__(self, *, encoder, outputs, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        if not isinstance(encoder, EmbeddingModel):
            raise TypeError("encoder must implement EmbeddingModel.")
        self.encoder = encoder
        self.outputs = {name: {"classes": list(output["classes"]), "sequence_len": int(output["sequence_len"])} for name, output in outputs.items()}
        if not self.outputs or any(not output["classes"] or output["sequence_len"] < 1 for output in self.outputs.values()):
            raise ValueError("outputs must define non-empty classes and a positive sequence_len for every task.")
        self.heads = torch.nn.ModuleDict({name: torch.nn.Linear(encoder.feature_dim(), output["sequence_len"] * len(output["classes"])) for name, output in self.outputs.items()})

    def compute(self, data):
        features = self.encoder.features(data)
        if features.ndim != 2 or features.shape[1] != self.encoder.feature_dim():
            raise ValueError(f"Expected encoder features [batch, {self.encoder.feature_dim()}], got {tuple(features.shape)}.")
        return {name: self.heads[name](features).view(features.shape[0], output["sequence_len"], len(output["classes"])) for name, output in self.outputs.items()}

    def input_spec(self):
        return self.encoder.input_spec()
