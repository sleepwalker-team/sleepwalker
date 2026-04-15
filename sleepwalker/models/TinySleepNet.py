"""TinySleepNet architecture used by the sleep-staging training script.

The repository uses this module as one of several interchangeable sleep-stage
classification backbones. It follows the broad CNN-plus-recurrent structure of
TinySleepNet while exposing repository-specific options such as different output
aggregation strategies and optional omission of the recurrent stage.
"""

from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from sleepwalker.models.Basemodel import BaseModel

class TinySleepNet(BaseModel):
    """Build a TinySleepNet-style model for sleep staging.

    Args:
        ts_len: Total input length in samples.
        n_channels: Number of input channels.
        classes: Optional class labels. When omitted, the model exposes
            features only.
        seq_len: Number of temporal chunks to split each input window into
            before applying the CNN front-end.
        sampling_frequency: Sampling frequency used to derive the first CNN
            filter and stride sizes.
        n_rnn_units: Hidden size of the optional recurrent stage.
        n_rnn_layers: Number of recurrent layers.
        use_lstm: Whether to apply the recurrent stage after the CNN front-end.
        output_strategy: How to reduce recurrent outputs. Current code supports
            ``flatten``, ``last``, ``center``, ``mean``, and ``sequence``.
        preprocessors: Optional preprocessors applied by ``BaseModel``.

    Notes:
        The file retains paper references from the original implementation, but
        this docstring describes repository behavior rather than claiming exact
        paper equivalence.
    """
    def __init__(
        self,
        *,
        ts_len,
        n_channels,
        classes=None,
        seq_len,
        sampling_frequency,
        n_rnn_units=128,
        n_rnn_layers=1,
        use_lstm=False,
        output_strategy = "flatten",
        preprocessors: Optional[Iterable] = None
    ):
        super().__init__(preprocessors)

        self.classes = list(classes) if classes is not None else None
        self.ts_len = ts_len
        self.n_channels = n_channels
        self.use_lstm = use_lstm
        self.seq_len = seq_len
        self.n_rnn_units = n_rnn_units
        self.n_rnn_layers = n_rnn_layers
        self.sampling_frequency = sampling_frequency

        if self.ts_len % self.seq_len != 0:
            raise ValueError(f"seq_len len (= {seq_len}) must be divisible by the total length of the input (= {self.ts_len})")

        first_filter_size = int(self.sampling_frequency / 2)
        first_filter_stride = int(self.sampling_frequency / 16)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.n_channels, out_channels=128, kernel_size=first_filter_size, stride=first_filter_stride, bias=False),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )

        with torch.no_grad():
            B, T, D = (1, self.ts_len, self.n_channels)
            x = torch.zeros(B, T, D)
            chunk_len = T // self.seq_len

            x = x.view(B, self.seq_len, chunk_len, D)        # (B, self.seq_len, chunk_len, D)
            x = x.permute(0, 1, 3, 2)                        # (B, self.seq_len, D, chunk_len)
            x = x.reshape(B * self.seq_len, D, chunk_len)    # (B * self.seq_len, D, chunk_len)

            #x = x.view(B, chunk_len, self.seq_len, D).permute(0,2,1,3).reshape(-1, chunk_len, D) # (B*S, chunk_len, D)
            #x = x.permute(0,2,1) # (B*S, D, chunk_len)
            self.cnn_output_size = self.cnn(x).shape[1]
        
        self.output_strategy = output_strategy
        if self.use_lstm:
            self._feature_dim = self.n_rnn_units * self.seq_len if self.output_strategy == "flatten" else self.n_rnn_units
        else:
            self._feature_dim = self.cnn_output_size
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.cnn_output_size,  
                hidden_size=self.n_rnn_units,
                num_layers=self.n_rnn_layers,
                batch_first=True,
                dropout=0.5 if self.n_rnn_layers > 1 else 0.0
            )
            if self.classes is not None:
                if self.output_strategy == "flatten":
                    self.fc = nn.Linear(self.n_rnn_units * self.seq_len, len(self.classes))
                else:
                    self.fc = nn.Linear(self.n_rnn_units, len(self.classes))
            else:
                self.fc = None
        else:
            self.fc = nn.Linear(self.cnn_output_size, len(self.classes)) if self.classes is not None else None

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN or CNN-plus-RNN features from input windows.

        Args:
            x: Input tensor shaped ``[batch, time, channels]``.

        Returns:
            Either a flattened feature tensor or a sequence of features,
            depending on ``use_lstm`` and ``output_strategy``.

        Raises:
            ValueError: If an unknown output strategy is configured.
        """
        # x: (B, T, D)
        B, T, D = x.shape
        chunk_len = T // self.seq_len

        x = x.view(B, self.seq_len, chunk_len, D)        # (B, self.seq_len, chunk_len, D)
        x = x.permute(0, 1, 3, 2)                        # (B, self.seq_len, D, chunk_len)
        x = x.reshape(B * self.seq_len, D, chunk_len)    # (B * self.seq_len, D, chunk_len)
        
        x = self.cnn(x)  # (B*S, F)

        if self.use_lstm:
            x = x.view(B, self.seq_len, self.cnn_output_size)  # (B, S, F)
            lstm_out, (hn, cn) = self.lstm(x)
            
            if self.output_strategy == "flatten":
                x = lstm_out.contiguous()
                x = x.view(B, -1)  
            elif self.output_strategy == "last":
                x = lstm_out[:, -1, :] 
            elif self.output_strategy == "center":
                x = lstm_out[:, self.seq_len // 2, :]  
            elif self.output_strategy == "mean":
                x = lstm_out.mean(dim=1)
            elif self.output_strategy == "sequence":
                x = lstm_out
            else:
                valid_strategies = ["flatten", "last", "center", "sequence"]
                raise ValueError(f"Unknown output strategy: {self.output_strategy}. Valid strategies are: {valid_strategies}")

        return x

    def feature_dim(self) -> int:
        """Return the feature size produced by :meth:`_features`."""
        return self._feature_dim

    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        """Project extracted features into class logits.

        Args:
            x: Feature tensor returned by :meth:`_features`.

        Returns:
            A class-logit tensor.

        Raises:
            ValueError: If the model was created without ``classes``.
        """
        if self.fc is None or self.classes is None:
            raise ValueError("TinySleepNet.classifier() requires classes to be set.")
        return self.fc(x)
