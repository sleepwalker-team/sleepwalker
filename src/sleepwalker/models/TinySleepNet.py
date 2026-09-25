"""TinySleepNet architecture used by the sleep-staging training script.

The repository uses this module as one of several interchangeable sleep-stage
classification backbones. It follows the broad CNN-plus-recurrent structure of
TinySleepNet while optionally omitting the recurrent stage.
"""

from typing import Optional
import torch
import torch.nn as nn

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel

class TinySleepNet(BaseModel, EmbeddingModel, ClassifierModel):
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
        preprocessors: Optional preprocessors applied by ``BaseModel``.
        sequence_len: Number of center-aligned chunks to return.

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
        preprocessors: Optional[list[nn.Module]] = None,
        sequence_len: int = 1,
    ):
        super().__init__(preprocessors=preprocessors)

        self.classes = list(classes) if classes is not None else None
        self.ts_len = ts_len
        self.n_channels = n_channels
        self.use_lstm = use_lstm
        self.seq_len = seq_len
        self.sequence_len = int(sequence_len)
        if self.sequence_len < 1:
            raise ValueError("sequence_len must be at least 1.")
        if self.sequence_len > self.seq_len:
            raise ValueError(f"sequence_len={self.sequence_len} exceeds seq_len={self.seq_len}.")
        self.n_rnn_units = n_rnn_units
        self.n_rnn_layers = n_rnn_layers
        self.sampling_frequency = sampling_frequency

        if self.ts_len % self.seq_len != 0:
            raise ValueError(f"ts_len={self.ts_len} must be divisible by seq_len={self.seq_len}.")

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
        
        output_feature_size = self.n_rnn_units if self.use_lstm else self.cnn_output_size
        self._feature_dim = self.sequence_len * output_feature_size
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.cnn_output_size,  
                hidden_size=self.n_rnn_units,
                num_layers=self.n_rnn_layers,
                batch_first=True,
                dropout=0.5 if self.n_rnn_layers > 1 else 0.0
            )
        self.fc = nn.Linear(output_feature_size, len(self.classes)) if self.classes is not None else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN or CNN-plus-RNN features from input windows.

        Args:
            x: Input tensor shaped ``[batch, time, channels]``.

        Returns:
            Center-aligned sequence features flattened to ``[B, S*E]``.
        """
        # x: (B, T, D)
        B, T, D = x.shape
        chunk_len = T // self.seq_len

        x = x.view(B, self.seq_len, chunk_len, D)        # (B, self.seq_len, chunk_len, D)
        x = x.permute(0, 1, 3, 2)                        # (B, self.seq_len, D, chunk_len)
        x = x.reshape(B * self.seq_len, D, chunk_len)    # (B * self.seq_len, D, chunk_len)
        
        x = self.cnn(x)  # (B*S, F)
        x = x.view(B, self.seq_len, self.cnn_output_size)

        if self.use_lstm:
            lstm_out, (hn, cn) = self.lstm(x)
            x = lstm_out

        sequence_start = (self.seq_len - self.sequence_len) // 2
        return x[:, sequence_start:sequence_start + self.sequence_len].reshape(B, -1)

    def feature_dim(self) -> int:
        """Return the feature size produced by :meth:`encode`."""
        return self._feature_dim

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, int | str]]:
        """Describe the raw BTC input shape expected by ``forward``."""
        return (
            (1, self.ts_len, self.n_channels),
            {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels},
        )

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed inputs and project them into class logits.

        Args:
            x: Preprocessed model input.

        Returns:
            A class-logit tensor.

        Raises:
            ValueError: If the model was created without ``classes``.
        """
        x = self.encode(x)
        if self.fc is None or self.classes is None:
            raise ValueError("TinySleepNet classification requires classes to be set.")
        feature_size = self.n_rnn_units if self.use_lstm else self.cnn_output_size
        expected = self.sequence_len * feature_size
        if x.ndim != 2 or x.shape[-1] != expected:
            raise ValueError(f"Expected TinySleepNet features shaped [B, {expected}], got {tuple(x.shape)}.")
        x = x.view(x.shape[0], self.sequence_len, feature_size)
        return self.fc(x)
