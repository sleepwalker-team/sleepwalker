from functools import partial

import torch

from sleepwalker.trainer.BaseTrainer import build_lr_scheduler


def make_optimizer():
    return torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.01)


def test_one_cycle_infers_epochs_and_steps_per_epoch():
    scheduler, per_batch = build_lr_scheduler(
        partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=0.1),
        make_optimizer(),
        epochs=3,
        steps_per_epoch=4,
    )

    assert scheduler.total_steps == 12
    assert per_batch is True


def test_one_cycle_preserves_explicit_total_steps():
    scheduler, per_batch = build_lr_scheduler(
        partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=0.1, total_steps=7),
        make_optimizer(),
        epochs=3,
        steps_per_epoch=4,
    )

    assert scheduler.total_steps == 7
    assert per_batch is True


def test_cyclic_and_linear_scheduler_step_modes():
    cyclic, cyclic_per_batch = build_lr_scheduler(
        partial(torch.optim.lr_scheduler.CyclicLR, base_lr=0.001, max_lr=0.01),
        make_optimizer(),
        epochs=3,
        steps_per_epoch=4,
    )
    linear, linear_per_batch = build_lr_scheduler(
        partial(torch.optim.lr_scheduler.LinearLR, total_iters=3),
        make_optimizer(),
        epochs=3,
        steps_per_epoch=4,
    )

    assert isinstance(cyclic, torch.optim.lr_scheduler.CyclicLR)
    assert cyclic_per_batch is True
    assert isinstance(linear, torch.optim.lr_scheduler.LinearLR)
    assert linear_per_batch is False
