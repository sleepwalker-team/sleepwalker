
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from sleepwalker.models.preprocessors.ChannelSampler import ChannelSampler
from sleepwalker.models.preprocessors.Crop import Crop
from sleepwalker.models.preprocessors.EmpiricalClipScaler import EmpiricalClipScaler
from sleepwalker.models.preprocessors.FIR import FIR
from sleepwalker.models.preprocessors.Normalize import Normalize
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from sleepwalker.models.preprocessors.Spectogram import Spectogram
from sleepwalker.models.preprocessors.ZNormalize import ZNormalize
from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.datasets.ABC import ABC
from sleepwalker.datasets import ChannelConfig
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split

from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer      

# The following classes are from old sleepwalker Utime

class ChannelWiseNormalization(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ChannelWiseNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x has shape (batch_size, num_channels, time_steps)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        x_scaled = self.gamma.view(1, -1, 1) * x_normalized + self.beta.view(1, -1, 1)
        return x_scaled


class Conv1dLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(Conv1dLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # Permute to [batch_size, length, channels] for LayerNorm
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        # Permute back to [batch_size, channels, length]
        x = x.permute(0, 2, 1)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()

        # Depthwise convolution: one filter per input channel
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )

        # Pointwise (1x1) convolution: mixes channels
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# UTime model definition abstract in forward function and modules used in forward function

class UTime(BaseModel):
    def __init__(self, forward_function, classes, n_channels, modules, preprocessors=None):
        super(UTime, self).__init__(preprocessors=preprocessors)
        self.forward_function = forward_function
        self.classes = list(classes)
        self.nchannel = n_channels
        self.nclass = len(classes)
        name = "module"
        for i, m in enumerate(modules):
            self.add_module(name + str(i), m)

    def _forward(self, x):
        return self.forward_function(x)     


# Here the interesting part begins: training a UTime model for ABC dataset

# Parameters for this run
edf_folder = "/Users/felixlaarmann/Desktop/Projekte/abc/polysomnography"
epochs = 5

if __name__ == "__main__":
    all_patients = get_edf_files_in_repo(edf_folder, recursive=True)
    train_patients, test_patients = random_split(all_patients, test_frac=0.1)       


    def build_loader(patients):
        data = ABC(annotator="nsrr",
            channels=[ChannelConfig(name='Sp02', normalizer=None), ChannelConfig(name='ECG1', normalizer=None), ChannelConfig(name='ECG2', normalizer=None), ChannelConfig(name='Thor', normalizer=None)],
            patients=patients, num_workers=8,
            sample_frequency=100, event_mapping={'hypopnea|hypopnea': 'hypopnea', 'central apnea|central apnea': 'apnea', 'obstructive apnea|obstructive apnea': 'apnea'},
            online_filtering=True, total_input="30s",
            target_resolution="1s",
            get_item=partial(MulticlassTrainer.get_item))
        sample = torch.utils.data.RandomSampler(data, replacement=True, num_samples=10)
        loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=sample is None, sampler=sample,
                                             num_workers=8, pin_memory=False, collate_fn=batch_collate,
                                             drop_last=False, persistent_workers=True)
        return loader, data
       

    train_loader, dataset = build_loader(train_patients)




    loss = torch.nn.functional.cross_entropy

    def forward(x): 

        x = x.swapaxes(1, 2)
        T = x.shape[-1]

#encoding


        c1 = nn.Conv1d(4, 512, kernel_size=2, stride=1, padding=0, dilation=1, bias=True)
        x = c1(x)
        norm = ChannelWiseNormalization(512, 0.001)
        x = norm(x)
        activation = nn.ReLU()
        x = activation(x)
        c2 = nn.Conv1d(512, 512, kernel_size=2, stride=1, padding=0, dilation=1, bias=True)
        x = c2(x)
        x = norm(x)
        x = activation(x)
        dropout = nn.Dropout1d(p=0.1)
        x = dropout(x)

        y0 = x
        mp = nn.MaxPool1d(5, stride=1, padding=0, dilation=1)
        x = mp(y0)
        if x.shape[-1] < 1:
            raise ValueError("Encoder output is empty after pooling. Reduce the pooling size or number of layers.")




        c1 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=0, dilation=1, bias=True)
        x = c1(x)
        norm = ChannelWiseNormalization(256, 0.001)
        x = norm(x)
        activation = nn.ReLU()
        x = activation(x)
        c2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=0, dilation=1, bias=True)
        x = c2(x)
        x = norm(x)
        x = activation(x)
        dropout = nn.Dropout1d(p=0.3)
        x = dropout(x)

        y1 = x
        mp = nn.MaxPool1d(3, stride=1, padding=0, dilation=1)
        x = mp(y1)
        if x.shape[-1] < 1:
            raise ValueError("Encoder output is empty after pooling. Reduce the pooling size or number of layers.")




        c1 = nn.Conv1d(256, 64, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        x = c1(x)
        norm = ChannelWiseNormalization(64, 0.001)
        x = norm(x)
        activation = nn.ReLU()
        x = activation(x)
        c2 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        x = c2(x)
        x = norm(x)
        x = activation(x)
        dropout = nn.Dropout1d(p=0.3)
        x = dropout(x)

        y2 = x
        mp = nn.MaxPool1d(5, stride=1, padding=0, dilation=1)
        x = mp(y2)
        if x.shape[-1] < 1:
            raise ValueError("Encoder output is empty after pooling. Reduce the pooling size or number of layers.")

# bottleneck

        c1 = nn.Conv1d(64, 64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        x = c1(x)
        norm = ChannelWiseNormalization(64, 0.001)
        x = norm(x)
        activation = nn.ReLU()
        x = activation(x)
        c2 = nn.Conv1d(64, 64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        x = c2(x)
        x = norm(x)
        x = activation(x)
        dropout = nn.Dropout1d(p=0.3)
        x = dropout(x)

# decoding

        up = nn.Upsample(scale_factor=5, mode="nearest")
        x = up(x)
        output_size = y2.size(2)

        if x.size(2) != output_size:
            diff = output_size - x.size(2)
            x = F.pad(x, (0, diff))  # Apply zero padding to the end of the dimension

        x = torch.cat([x, y2], dim=1)

        c1 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        x = c1(x)
        norm = ChannelWiseNormalization(256, 0.001)
        x = norm(x)
        activation = nn.ReLU()
        x = activation(x)
        c2 = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        x = c2(x)
        x = norm(x)
        x = activation(x)
        dropout = nn.Dropout1d(p=0.3)
        x = dropout(x)




        up = nn.Upsample(scale_factor=3, mode="nearest")
        x = up(x)
        output_size = y1.size(2)

        if x.size(2) != output_size:
            diff = output_size - x.size(2)
            x = F.pad(x, (0, diff))  # Apply zero padding to the end of the dimension

        x = torch.cat([x, y1], dim=1)

        c1 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1, bias=True)
        x = c1(x)
        norm = ChannelWiseNormalization(512, 0.001)
        x = norm(x)
        activation = nn.ReLU()
        x = activation(x)
        c2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1, bias=True)
        x = c2(x)
        x = norm(x)
        x = activation(x)
        dropout = nn.Dropout1d(p=0.3)
        x = dropout(x)




        up = nn.Upsample(scale_factor=5, mode="nearest")
        x = up(x)
        output_size = y0.size(2)

        if x.size(2) != output_size:
            diff = output_size - x.size(2)
            x = F.pad(x, (0, diff))  # Apply zero padding to the end of the dimension

        x = torch.cat([x, y0], dim=1)

        c1 = nn.Conv1d(1024, 4, kernel_size=2, stride=1, padding=0, dilation=1, bias=True)
        x = c1(x)
        norm = ChannelWiseNormalization(4, 0.001)
        x = norm(x)
        activation = nn.ReLU()
        x = activation(x)
        c2 = nn.Conv1d(4, 4, kernel_size=2, stride=1, padding=0, dilation=1, bias=True)
        x = c2(x)
        x = norm(x)
        x = activation(x)
        dropout = nn.Dropout1d(p=0.1)
        x = dropout(x)


        final_conv = nn.Conv1d(4, 512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        x = final_conv(x)
        x = x.mean(dim=2)
        mlp = nn.Linear(512, 2, bias=False)
        x = mlp(x)
        return x


    modules = [
        nn.Conv1d(4, 512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True), 
        nn.Linear(512, 2, bias=False), 
        nn.MaxPool1d(5, stride=1, padding=0, dilation=1), 
        nn.ReLU(), 
        nn.Dropout1d(p=0.1), 
        nn.Conv1d(4, 512, kernel_size=2, stride=1, padding=0, dilation=1, bias=True), 
        nn.Conv1d(512, 512, kernel_size=2, stride=1, padding=0, dilation=1, bias=True), 
        ChannelWiseNormalization(512, 0.001), 
        nn.Upsample(scale_factor=5, mode="nearest"), 
        nn.ReLU(), 
        nn.Dropout1d(p=0.1), 
        nn.Conv1d(1024, 4, kernel_size=2, stride=1, padding=0, dilation=1, bias=True), 
        nn.Conv1d(4, 4, kernel_size=2, stride=1, padding=0, dilation=1, bias=True), 
        ChannelWiseNormalization(4, 0.001), 
        nn.MaxPool1d(3, stride=1, padding=0, dilation=1), 
        nn.ReLU(), 
        nn.Dropout1d(p=0.3), 
        nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=0, dilation=1, bias=True), 
        nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=0, dilation=1, bias=True), 
        ChannelWiseNormalization(256, 0.001), 
        nn.Upsample(scale_factor=3, mode="nearest"), 
        nn.ReLU(), 
        nn.Dropout1d(p=0.3), 
        nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1, bias=True), 
        nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1, bias=True), 
        ChannelWiseNormalization(512, 0.001), 
        nn.MaxPool1d(5, stride=1, padding=0, dilation=1), 
        nn.ReLU(), 
        nn.Dropout1d(p=0.3), 
        nn.Conv1d(256, 64, kernel_size=5, stride=1, padding=0, dilation=1, bias=True), 
        nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0, dilation=1, bias=True), 
        ChannelWiseNormalization(64, 0.001), 
        nn.Upsample(scale_factor=5, mode="nearest"), 
        nn.ReLU(), 
        nn.Dropout1d(p=0.3), 
        nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=0, dilation=1, bias=True), 
        nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0, dilation=1, bias=True), 
        ChannelWiseNormalization(256, 0.001), 
        nn.ReLU(), 
        nn.Dropout1d(p=0.3), 
        nn.Conv1d(64, 64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True), 
        nn.Conv1d(64, 64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True), 
        ChannelWiseNormalization(64, 0.001)
        ]

    model = UTime(forward, dataset.get_classes(), len(dataset.channels), modules, preprocessors=(ChannelSampler(4), ))  # n_channel = len(dataset.channels)?
                

    trainer = MulticlassTrainer(
                epochs=epochs,
                optimizer=lambda m: torch.optim.Adam(params=m.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-10, weight_decay=0, amsgrad=True),
                lr_scheduler=lambda opti: torch.optim.lr_scheduler.LinearLR(optimizer=opti, start_factor=1, end_factor=0.01, total_iters=50, last_epoch=-1),
                classes=dataset.get_classes(),
                save_every=10,
                loss_function=loss,
                device="cpu"
            )

    losses, cms = trainer.fit(model, train_loader)

    test_loader, _ = build_loader(test_patients)
    test_loss, test_cm = trainer.test(model, test_loader)

    record = {
                "test_loss": test_loss,
                "test_cm": test_cm,
                "train_loss": losses,
                "train_cm": cms,
                "classes": dataset.get_classes(),
             }

    print(record)
