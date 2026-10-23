"""A self-contained frozen feature model followed by a native batch estimator."""

import numpy as np
import torch

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel


class BatchClassifierModel(BaseModel, ClassifierModel, EmbeddingModel):
    def __init__(self, embedding_model, batch_model, classes):
        super().__init__(preprocessors=[])
        if not isinstance(embedding_model, torch.nn.Module) or not isinstance(embedding_model, EmbeddingModel):
            raise TypeError("embedding_model must be an embedding-capable torch module.")
        self.embedding_model = embedding_model
        self.batch_model = batch_model
        self.classes = list(classes)
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must contain at least two unique labels.")
        if list(batch_model.classes) != self.classes:
            raise ValueError("Batch model classes must match the configured class order.")
        self.embedding_model.requires_grad_(False)
        self.embedding_model.eval()

    def validate_fitted(self):
        if not self.batch_model.is_fitted():
            raise RuntimeError("The batch estimator has not been fitted.")

    def train(self, mode=True):
        super().train(mode)
        self.embedding_model.eval()
        return self

    def encode(self, x):
        self.embedding_model.eval()
        with torch.no_grad():
            features = self.embedding_model.features(x)
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape != (x.shape[0], self.feature_dim()) or not torch.isfinite(features).all():
            raise ValueError("The embedding model must return one fixed-dimension vector per input row.")
        return features

    def compute(self, x):
        self.validate_fitted()
        features = self.encode(x)
        probabilities = np.asarray(self.batch_model.predict_proba(features.detach().cpu().to(torch.float32).numpy()))
        if probabilities.shape != (len(x), len(self.classes)):
            raise ValueError("Estimator returned an invalid probability matrix shape.")
        order = list(self.batch_model.probability_classes())
        if len(order) != len(self.classes) or set(order) != set(self.classes):
            raise ValueError("Estimator probability columns must cover the declared classes exactly.")
        probabilities = probabilities[:, [order.index(label) for label in self.classes]]
        if probabilities.shape != (len(x), len(self.classes)) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Estimator returned invalid class probabilities.")
        if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Estimator probabilities must sum to one for each sample.")
        values = torch.as_tensor(probabilities, device=x.device, dtype=torch.float32)
        return values.clamp_min(torch.finfo(values.dtype).tiny).log().unsqueeze(1)

    def feature_dim(self):
        return self.embedding_model.feature_dim()

    def input_spec(self):
        return self.embedding_model.input_spec()
