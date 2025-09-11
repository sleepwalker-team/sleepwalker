import torch
from sleepwalker.trainer import Trainer

class MulticlassTrainer(Trainer):

    def train_step(self, model: torch.nn.Module, batch):
        x = batch['data']
        y = batch['target']
        logits = model(x)
        try:
            loss = model.loss.loss_function(logits, y)
        except AttributeError:
            loss = model.module.loss.loss_function(logits, y)
        return loss

    def val_step(self, model: torch.nn.Module, batch):
        x = batch['data']
        y = batch['target']
        logits = model(x)
        try:
            loss = model.loss.loss_function(logits, y)
        except AttributeError:
            loss = model.module.loss.loss_function(logits, y)
        return loss