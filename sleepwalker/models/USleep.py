import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from sleepwalker.utils import logger

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, dilation, padding):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if isinstance(activation, tuple):
            self.a1 = nn.ELU() if activation[0] == 'elu' else nn.Tanh() if activation[0] == "tanh" else nn.ReLU()
            self.a2 = nn.ELU() if activation[1] == 'elu' else nn.Tanh() if activation[1] == "tanh" else nn.ReLU()
        else:
            self.a1 = nn.ELU() if activation == 'elu' else nn.Tanh() if activation == "tanh" else nn.ReLU()
            self.a2 = nn.ELU() if activation == 'elu' else nn.Tanh() if activation == "tanh" else nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.a1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.a2(x)
        return x

class USleep(BaseModel):
    """
    U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging by Perslev et al. in NeurIPs, 2019
    Code: https://github.com/perslev/U-Time

    Model size: 1.2 Mio (paper) vs. 1.8 Mio parameters (this config)

    Expected performance (sleep-edf-39 / sleep-edf-153) in the paper:
    - Accuracy:     -
    - Macro F1:     0.79 / 0.76
    - Cohens Kappa: -

    Summary of differences to the paper:
      - Resampling: We use nearest neighbor resampling, whereas the paper uses scipy.signal.resample_poly
      - RobustScaler: We use an online implementation of RobustScaler, whereas the paper uses sklearn.preprocessing.RobustScaler
      - (IMPORTANT) This UTime/Usleep architecture does _not_ fully match the original paper! This implementation is more general and allows for a very flexible configuration of the structure of the network. Arguably, this has nothing to do with the original model anymore. 
    """
    def __init__(self, 
        *,
        ts_len, 
        n_channels,
        classes,
        sampling_frequency,
        depth=4,
        init_filters=16,
        kernel_size=5,
        dilation=1,
        epoch_len="30s",
        output_strategy='mean', 
        activation="relu",
        ):
        super().__init__(preprocessors=[RobustScaler()])

        self.classes = classes
        self.ts_len = ts_len
        self.n_channels = n_channels
        self.sampling_frequency = sampling_frequency

        if depth >= 5:
            depth = 4
            logger.warning("Depth >= 5 is not supported by the original U-Time configuration. Restricting to depth = 4.")

        self.output_strategy = output_strategy
        self.epoch_len_str = epoch_len

        self.epoch_len_s = pd.Timedelta(epoch_len).total_seconds()
        if self.epoch_len_s <= 0:
            raise ValueError("Epoch length must be positive.")

        self.samples_per_epoch = int(self.epoch_len_s * self.sampling_frequency)
        if self.samples_per_epoch <= 0:
            raise ValueError(f"Epoch length '{epoch_len}' and sampling rate '{self.sampling_rate}' result in non-positive number of samples per epoch.")

        self.depth = depth
        self.pools = [10, 8, 6, 4][:depth]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.residuals = []

        in_ch = self.n_channels
        filters = init_filters
        self.encoder_channels = []

        for d in range(depth):
            self.encoder.append(ConvBlock(in_ch, filters, kernel_size, activation, dilation, padding=kernel_size//2))
            self.encoder_channels.append(filters)
            in_ch = filters
            filters *= 2

        self.bottom_channels = filters
        self.bottom = ConvBlock(in_ch, self.bottom_channels, kernel_size, activation, 1, padding=kernel_size//2)

        for i, ch in enumerate(reversed(self.encoder_channels)):
            if i == self.depth - 1:
                self.decoder.append(ConvBlock(in_channels=self.bottom_channels + ch, out_channels=ch,
                                    kernel_size=kernel_size, activation=(activation, "tanh"),
                                    dilation=1, padding=kernel_size//2))
            else:
                self.decoder.append(ConvBlock(in_channels=self.bottom_channels + ch, out_channels=ch,
                                    kernel_size=kernel_size, activation=activation,
                                    dilation=1, padding=kernel_size//2))
            self.bottom_channels = ch

        self.final_conv = nn.Conv1d(self.bottom_channels, len(self.classes), kernel_size=1)
        self.avg_pool = nn.AvgPool1d(kernel_size=self.samples_per_epoch, stride=self.samples_per_epoch)
        self.act = nn.ELU() if activation == 'elu' else nn.Tanh() if activation == "tanh" else nn.ReLU()
        if self.output_strategy == "flatten":
            N = self.ts_len // self.samples_per_epoch
            self.fc = nn.Linear(N*len(self.classes), len(self.classes))

    def _forward(self, x: torch.Tensor) -> torch.Tensor: 
        B, T, D = x.shape
        if T % self.samples_per_epoch != 0:
            raise ValueError(f"Input time axis T={T} is not divisible by samples_per_epoch={self.samples_per_epoch}. Ensure input length matches the expected epoch segmentation.")

        N = T // self.samples_per_epoch

        x = x.view(B, N, self.samples_per_epoch, D).permute(0,1,3,2).reshape(B*N, D, self.samples_per_epoch)
        self.residuals = []

        # Encoder
        for enc, p in zip(self.encoder, self.pools):
            x = enc(x)
            self.residuals.append(x)
            x = F.max_pool1d(x, kernel_size=p)

        # Bottom
        x = self.bottom(x)

        # Decoder
        for dec, res, p in zip(self.decoder, reversed(self.residuals), reversed(self.pools)):
            x = F.interpolate(x, scale_factor=p, mode='nearest')
            if x.shape[-1] != res.shape[-1]:
                diff = res.shape[-1] - x.shape[-1]
                x = F.pad(x, (0, diff))
            x = torch.cat([res, x], dim=1)
            x = dec(x)

        # Final classifier
        x = self.final_conv(x)                          # (B*N, C, T)
        x = self.avg_pool(x)                            # (B*N, C, 1)
        x = x.squeeze(-1).view(B, N, len(self.classes))  # (B, N, C)

        if self.output_strategy == 'mean':
            x = x.mean(dim=1) 
        elif self.output_strategy == 'center':
            x = x[:, x.shape[2] // 2, :]
        elif self.output_strategy == 'last':
            x = x[:, -1, :]
        elif self.output_strategy == 'flatten':
            x = x.view(B, -1)
            x = self.fc(x)
        elif self.output_strategy == "sequence":
            x = x.reshape(B*N, -1)
        else:
            raise ValueError(f"Unknown temporal reduction mode: {self.output_strategy}")
    
        return x
        