from sleepwalker.trainer.tasks.sleep_staging import DATASET_CFG


def test_grouped_channels_are_part_of_dataset_defaults():
    cfg = DATASET_CFG["sleepedfx"]

    assert [c.name for c in cfg["channels"]] == ["EEG Fpz-Cz"]
    assert [c.name for c in cfg["grouped_channels"]] == ["EEG Fpz-Cz", "EEG Pz-Oz"]
    assert all(c.group == "eeg" for c in cfg["grouped_channels"])


def test_target_classes_are_derived_from_event_mapping():
    classes = sorted(set(DATASET_CFG["sleepedfx"]["event_mapping"].values()))

    assert classes == ["n1", "n2", "n3", "rem", "wake"]
