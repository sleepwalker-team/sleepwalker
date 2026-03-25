import numpy as np
from sklearn.metrics import confusion_matrix
import torch

from sleepwalker.datasets.utils import RepeatSampler
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.utils import logger

from sleepwalker.trainer.utils import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix

import torch.nn as nn
from torch.autograd import Function

# --------------------------
# Gradient Reversal Function
# --------------------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# GroupedChannel
class GradReverseTrainer(MulticlassTrainer):
    def __init__(
        self,
        feature_dim,
        n_domains,
        lambda_domain=0.2,            # weight of domain loss
        domain_hidden=128,            # small hidden layer
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lambda_domain = lambda_domain
        self.domain_hidden = domain_hidden

        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, self.domain_hidden),
            nn.ReLU(),
            nn.Linear(self.domain_hidden, n_domains)
        ).to(self.device)
        self.n_domains = n_domains

    def run_epoch(self, loader, opt, model, prefix=""):
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)
        nc = self.num_classes
        
        loss_sum = 0
        cm_sum = np.zeros((nc, nc), dtype=np.int64)
        domain_loss_sum = 0
        cnt = 0

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        n_repeat = loader.sampler.n_repeat if isinstance(getattr(loader, "sampler", None), RepeatSampler) else 1

        for batch in loader:
            x = batch["data"].to(self.device)
            y = batch["target"].to(self.device)
            domains = torch.tensor(batch["dataset"], dtype=torch.long, device=self.device)

            if opt is not None:
                x = self.apply_train_transform(x)

            if n_repeat > 1:
                if x.shape[0] % n_repeat != 0:
                    raise ValueError(f"Batch size {x.shape[0]} is not divisible by n_repeat={n_repeat}.")
                base_batch = x.shape[0] // n_repeat
                x_grouped = x.view(base_batch, n_repeat, *x.shape[1:])
                y = y.view(base_batch, n_repeat, *y.shape[1:])[:, 0]
                domains = domains.view(base_batch, n_repeat)[:, 0]
            else:
                base_batch = x.shape[0]

            if opt is not None: 
                opt.zero_grad(set_to_none=True)
            
            if n_repeat > 1:
                feats_sum = None
                logits_sum = None
                for repeat_idx in range(n_repeat):
                    current_x = model.apply_preprocessors(x_grouped[:, repeat_idx], len(model.preprocessors) + 1)
                    current_feats = model.features(current_x)
                    current_logits = model.classifier(current_feats)
                    feats_sum = current_feats if feats_sum is None else feats_sum + current_feats
                    logits_sum = current_logits if logits_sum is None else logits_sum + current_logits
                feats = feats_sum / n_repeat
                logits = logits_sum / n_repeat
            else:
                x = model.apply_preprocessors(x, len(model.preprocessors) + 1)
                feats = model.features(x)
                logits = model.classifier(feats)

            loss_task = self.loss_function(logits, y)

            if mode == "train":
                feats_rev = grad_reverse(feats, lambd=self.lambda_domain)
                d_logits = self.domain_head(feats_rev)
                loss_domain = nn.functional.cross_entropy(d_logits, domains)
            else:
                loss_domain = torch.tensor(0.0, device=self.device)

            loss = loss_task + loss_domain
            
            target_np = y.argmax(axis=1).cpu().numpy()
            pred_np = logits.argmax(axis=1).cpu().numpy()

            if opt is not None:
                loss.backward()
                opt.step()

            cm = confusion_matrix(target_np, pred_np, labels=range(len(self.classes)))
            cm_sum += cm
            cnt += 1
            loss_sum += float(loss.item())
            domain_loss_sum += float(loss_domain.item())

            step = self.steps[mode]
            self._log_from_cm(cm, float(loss.item()), mode=mode, scope="batch", step=step) 

            accs = cm_sum.trace() / cm_sum.sum() * 100.0
            f1_micro = f1_score_from_confusion_matrix(cm_sum, macro=False)
            f1_macro = f1_score_from_confusion_matrix(cm_sum, macro=True)
            coehns_kappa = cohen_kappa_from_confusion_matrix(cm_sum)

            desc = f"{prefix:<12} {loss_sum/cnt:2.4f} acc {accs:2.3f} " \
                   f"f1 (mi/ma) {f1_micro:1.4f}/{f1_macro:1.4f} κ {coehns_kappa:2.3f}" \
                   f"DL {domain_loss_sum/cnt:1.4f}"
            
            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1) 
        self._log_from_cm(cm_sum, epoch_loss, mode=mode, scope="epoch", step=self.epoch_step)  

        return epoch_loss, cm_sum  

    # def run_epoch(self, loader, opt, model, prefix=""):
    #     logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)
    #     nc = self.num_classes
        
    #     loss_sum = 0
    #     domain_loss_sum = 0   # track domain loss
    #     cm_sum = np.zeros((nc, nc), dtype=np.int64)
    #     cnt = 0

    #     mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
    #     if mode == "train":
    #         n_repeat = self.n_repeat_train
    #     else:
    #         n_repeat = self.n_repeat_test

    #     for batch in loader:
    #         y = batch["target"].to(self.device)
    #         domains = torch.tensor(batch["dataset"], dtype=torch.long, device=self.device)

    #         if opt is not None:
    #             opt.zero_grad(set_to_none=True)

    #         logits_sum = 0
    #         feat_sum = 0

    #         if self.groups:
    #             for i in range(n_repeat):
    #                 tmp_x = torch.stack([
    #                     x.sample(self.rng, self.groups).to(self.device)
    #                     for x in batch["data"]
    #                 ])  # (B, T, C)

    #                 # model must provide feature extraction and classification separately
    #                 feats = model.features(tmp_x)  # shape (B, F)
    #                 logits_i = model.classifier(feats)    # shape (B, nc)

    #                 logits_sum += logits_i
    #                 feat_sum += feats
    #             feats = feat_sum / n_repeat
    #             logits = logits_sum / n_repeat
    #         else:
    #             x = batch["data"].to(self.device)
    #             feats = model.features(x)
    #             logits = model.classifier(feats)

    #         loss_task = self.loss_function(logits, y)

    #         if mode == "train":
    #             # Gradient reversal
    #             feats_rev = grad_reverse(feats, lambd=self.lambda_domain)
    #             d_logits = self.domain_head(feats_rev)
    #             loss_domain = nn.functional.cross_entropy(d_logits, domains)
    #         else:
    #             loss_domain = torch.tensor(0.0, device=self.device)

    #         loss = loss_task + loss_domain

    #         # Compute predictions for stats
    #         target_np = y.argmax(axis=1).cpu().numpy()
    #         pred_np = logits.argmax(axis=1).cpu().numpy()

    #         if opt is not None:
    #             loss.backward()
    #             opt.step()

    #         cm = confusion_matrix(target_np, pred_np, labels=range(len(self.classes)))
    #         cm_sum += cm
    #         cnt += 1
    #         loss_sum += float(loss_task.item())
    #         domain_loss_sum += float(loss_domain.item())

    #         step = self.steps[mode]
    #         self._log_from_cm(cm, float(loss_task.item()), mode=mode, scope="batch", step=step) 

    #         accs = cm_sum.trace() / cm_sum.sum() * 100.0
    #         f1_micro = f1_score_from_confusion_matrix(cm_sum, macro=False)
    #         f1_macro = f1_score_from_confusion_matrix(cm_sum, macro=True)
    #         coehns_kappa = cohen_kappa_from_confusion_matrix(cm_sum)

    #         desc = (
    #             f"{prefix:<12} "
    #             f"{loss_sum/cnt:2.4f} "
    #             f"acc {accs:2.3f} "
    #             f"f1 (mi/ma) {f1_micro:1.4f}/{f1_macro:1.4f} "
    #             f"κ {coehns_kappa:2.3f} "
    #             f"DL {domain_loss_sum/cnt:1.4f}"
    #         )
            
    #         self.steps[mode] += 1
    #         logger.progress_status(desc)
    #         logger.progress_advance(loader.batch_size)

    #     logger.progress_close()
    #     epoch_loss = loss_sum / max(cnt, 1)
    #     self._log_from_cm(cm_sum, epoch_loss, mode=mode, scope="epoch", step=self.epoch_step)  

    #     return epoch_loss, cm_sum

    def fit(self, model, train_loader, val_loader = None):
        # TODO check model and loaders
        # TODO add annealing over epochs
        return super().fit(model, train_loader, val_loader)
