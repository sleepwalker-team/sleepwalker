from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from sleepwalker.models.Basemodel import BaseModel

class TinySleepNet(BaseModel):
    """
    Paper: TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring based on Raw Single-Channel EEG by Supratak et al. in Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), 2020
    Code: https://github.com/akaraspt/tinysleepnet

    Model size: 1.3 Mio (paper) vs. 1.1 Mio parameters (this config)

    Expected performance (sleep-edf-20 / sleep-edf-78) in the paper:
    - Accuracy:     85.5 / 83.1
    - Macro F1:     0.805 / 0.781
    - Cohens Kappa: 0.80 / 0.77

    Notes on the paper:
    - Depending on the configuration, the original code uses a low-pass filter on the data (see: https://github.com/akaraspt/tinysleepnet/blob/70f45cff92ac0bf0a718e522bc27cca2c63ceff7/train.py#L183). This is not mention in the paper and we do not use it here as well.
    - Data Augmentation works slightly different: The original paper uses two data augmentations thats a) shift the time dimension and b) skip entire 30s epochs. We combine both here into one augmentation by implementing comparably large shifts in the raw signals that potentially skip up to two epochs (i.e. shift by 60 s). 
    """
    def __init__(
        self,
        *,
        ts_len,
        n_channels,
        classes,
        seq_len,
        sampling_frequency,
        n_rnn_units=128,
        n_rnn_layers=1,
        use_lstm=False,
        output_strategy = "flatten",
        preprocessors: Optional[Iterable] = None
    ):
        super().__init__(preprocessors)

        self.classes = classes
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
            self.lstm = nn.LSTM(
                input_size=self.cnn_output_size,  
                hidden_size=self.n_rnn_units,
                num_layers=self.n_rnn_layers,
                batch_first=True,
                dropout=0.5 if self.n_rnn_layers > 1 else 0.0
            )
            if self.output_strategy == "flatten":
                self.fc = nn.Linear(self.n_rnn_units * self.seq_len, len(self.classes))
            else:
                self.fc = nn.Linear(self.n_rnn_units, len(self.classes))
        else:
            self.fc = nn.Linear(self.cnn_output_size, len(self.classes))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
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
                x = lstm_out.reshape(B * self.seq_len, -1)
            else:
                valid_strategies = ["flatten", "last", "center", "sequence"]
                raise ValueError(f"Unknown output strategy: {self.output_strategy}. Valid strategies are: {valid_strategies}")

        out = self.fc(x)
        if self.use_lstm and self.output_strategy == "sequence":
            out = out.view(B, self.seq_len, len(self.classes))
        return out #(out, x) if return_intermediate else out
