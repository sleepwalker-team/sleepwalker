import pandas as pd
import torch

from sleepwalker.trainer.utils.targets import prepare_multiclass_target


def make_target(columns: dict[str, list[float]], freq: str = "30s") -> pd.DataFrame:
    periods = len(next(iter(columns.values()))) if len(columns) > 0 else 4
    index = pd.date_range("2020-01-01", periods=periods, freq=freq)
    return pd.DataFrame(columns, index=index)


def test_prepare_multiclass_target_uses_fallback_when_positive_column_is_absent():
    target = make_target({"noisy": [0.0, 0.0, 0.0, 0.0]})

    item = prepare_multiclass_target(
        target,
        target_classes=["clean", "noisy"],
    )

    assert torch.equal(item["target"], torch.tensor([1.0, 0.0]))


def test_prepare_multiclass_target_builds_independent_target_extra():
    target = make_target(
        {
            "desaturation": [0.0, 0.0, 0.0, 0.0],
            "n2": [1.0, 1.0, 1.0, 1.0],
        }
    )
    target_extra = make_target({"desaturation": [1.0, 1.0, 0.0, 0.0]})

    item = prepare_multiclass_target(
        target,
        target_extra=target_extra,
        target_classes=["desaturation", "no desaturation"],
        filters=[{"columns": ["n1", "n2", "n3", "rem"], "percentage": 0.5, "mode": "min"}],
    )

    assert torch.equal(item["target"], torch.tensor([0.0, 1.0]))
    assert torch.equal(item["target_extra"], torch.tensor([1.0, 0.0]))


def test_prepare_multiclass_target_rejects_filter_columns():
    target = make_target(
        {
            "supine": [1.0, 1.0, 1.0, 1.0],
            "movement": [0.0, 1.0, 0.0, 0.0],
        }
    )

    item = prepare_multiclass_target(
        target,
        target_classes=["supine", "left", "right", "prone", "upright"],
        filters=[{"columns": ["unknown", "movement"], "percentage": 0.0, "mode": "max"}],
    )

    assert item is None


def test_prepare_multiclass_target_ignores_ambiguous_target_extra():
    target = make_target({"supine": [1.0, 1.0, 1.0, 1.0]})
    target_extra = make_target(
        {
            "supine": [1.0, 1.0, 0.0, 0.0],
            "left": [1.0, 1.0, 0.0, 0.0],
        }
    )

    item = prepare_multiclass_target(
        target,
        target_extra=target_extra,
        target_classes=["supine", "left", "right", "prone", "upright"],
    )

    assert torch.equal(item["target"], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert "target_extra" not in item


def test_prepare_multiclass_target_rejects_min_filter_when_sleep_is_missing():
    target = make_target({"desaturation": [0.0, 0.0, 0.0, 0.0]})

    item = prepare_multiclass_target(
        target,
        target_classes=["desaturation", "no desaturation"],
        filters=[{"columns": ["n1", "n2", "n3", "rem"], "percentage": 0.5, "mode": "min"}],
    )

    assert item is None
