from typing import Literal, Optional
import numpy as np
import torch


def dice_loss(pred, target, weight = None, epsilon=1e-3):
    """
    Dice loss for multi-class targets.
    Args:
        pred (torch.Tensor): Logits of shape (B, C).
        target (torch.Tensor): Class indices of shape (B,).
        weight (torch.Tensor): Class weights of shape (C,).
        epsilon (float): Small constant to avoid division by zero.
    Returns:
        torch.Tensor: Scalar Dice loss.
    """
    pred = torch.softmax(pred, dim=1)  # (B, C)
    #target_onehot = torch.nn.functional.one_hot(target.long(), num_classes=pred.shape[1]).float()  # (B, C)

    intersection = (pred * target).sum(dim=0)
    union = pred.sum(dim=0) + target.sum(dim=0)

    dice = (2 * intersection + epsilon) / (union + epsilon)  # (C,)

    if weight is not None:
        dice = dice * weight

    return 1 - dice.mean()

def class_weights_for_loss(user_weights: dict[str, float], class_distribution: dict[str, float], mode:Literal['inverse', 'inverse-log'] = "inverse") -> dict[str, float]:
    new_weights = {}

    if mode == "inverse":
        # Weight classes by their (inverse) occurrence. This can lead to relatively small losses,
        # hence we will also weight normalize it. This is technically not necessary.
        total_sum = 0
        for c in class_distribution.keys():
            new_weights[c] = user_weights.get(c, 1.0) * np.clip(1.0 / class_distribution[c], min = 1e-4)
            total_sum += new_weights[c]
        
        new_weights = {k:v/total_sum for k,v in new_weights.items()}
    else:
        """
        See 
            - MRASleepNet: a multi-resolution attention network for sleep stage classification using single-channel EEG by Rui Yu, Zhuhuang Zhou, Shuicai Wu, Xiaorong Gao and Guangyu Bin in Journal of Neural Engineering 2022, https://github.com/YuRui8879/MRASleepNet/blob/781aee2d2ff1422c598b099081a4c1d7d4bd05d7/DataAdapter/DataAdapter.py#L110
            - An Attention-Based Deep Learning Approach for Sleep Stage Classification With Single-Channel EEG by Eldele et al. in IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING 2021, https://github.com/emadeldeen24/AttnSleep/blob/6b4d2665884628c8a7bb09f36589a8ec0992f8e2/utils/util.py#L62
        """

        total = sum(class_distribution.values())
        factor = 1.0 / len(class_distribution)
        for c in class_distribution.keys():
            mu = factor * user_weights.get(c,1.0)
            new_weights[c] = mu * np.clip(np.log( (total * mu) / class_distribution[c]), min=1.0)
        
    return new_weights
