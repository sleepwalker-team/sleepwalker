from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "iclr2026" / "configs"
TASKS = ("sleep", "arousal", "breathing", "desaturation")
MODELS = ("sleepwalker", "osf", "sleepfm")


def read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_paper_config_layout_is_small_and_explicit():
    train = sorted((CONFIG_ROOT / "experts" / "train").glob("*.yml"))
    test = sorted((CONFIG_ROOT / "experts" / "test").glob("*.yml"))
    expected = {f"{task}_{model}" for task in TASKS for model in MODELS} | {"multitask_sleepwalker"}

    assert {path.stem for path in train} == expected
    assert {path.stem for path in test} == expected
    assert {path.name for path in (CONFIG_ROOT / "main").glob("*.yml")} == {"latent.yml", "probability.yml", "probability_osf.yml", "probability_sleepfm.yml", "sei.yml", "test.yml"}
    assert not (CONFIG_ROOT / "generate_configs.py").exists()
    assert not (CONFIG_ROOT / "train").exists()
    assert not (CONFIG_ROOT / "test").exists()
    assert (CONFIG_ROOT / "hsp.yml").is_file()
    assert (CONFIG_ROOT / "split.yml").is_file()
    assert not (CONFIG_ROOT / "mirror.yaml").exists()


def test_expert_train_configs_are_resolved_and_use_fixed_class_counts():
    package_paths = set()
    for path in sorted((CONFIG_ROOT / "experts" / "train").glob("*.yml")):
        config = read_config(path)
        assert config["run"]["n_samples"] == 250000
        assert config["run"]["patients_per_epoch"] == 512
        assert config["run"]["package_path"].startswith("iclr2026/models/")
        package_paths.add(config["run"]["package_path"])
        assert "patient_group_size" not in config["run"]
        assert "channel_catalog" not in config["data"]
        assert all("source" not in channel for channel in config["data"]["channels"])

        if path.stem == "multitask_sleepwalker":
            counts = config["trainer"]["task_config"]
            assert all(task["class_counts"] for task in counts.values())
            continue

        assert config["trainer"]["class_counts"]
        is_sleepwalker = path.stem.endswith("_sleepwalker")
        assert config["trainer"]["epochs"] == (100 if is_sleepwalker else 10)
        if is_sleepwalker:
            assert config["trainer"]["lr_scheduler"]["name"] == "torch.optim.lr_scheduler.OneCycleLR"
            assert config["trainer"]["save_every"] == 10
        else:
            assert "lr_scheduler" not in config["trainer"]
            assert config["trainer"]["save_every"] == 2
            assert config["model"]["head"] == "linear"
            assert config["model"]["freeze_encoder"] is True
    assert len(package_paths) == len(TASKS) * len(MODELS) + 1


def test_task_class_counts_match_across_expert_variants():
    for task in TASKS:
        sleepwalker = read_config(CONFIG_ROOT / "experts" / "train" / f"{task}_sleepwalker.yml")
        for foundation in ("osf", "sleepfm"):
            probe = read_config(CONFIG_ROOT / "experts" / "train" / f"{task}_{foundation}.yml")
            assert sleepwalker["trainer"]["class_counts"] == probe["trainer"]["class_counts"]


def test_multitask_config_reuses_the_single_task_class_counts():
    multitask = read_config(CONFIG_ROOT / "experts" / "train" / "multitask_sleepwalker.yml")
    task_blocks = [
        multitask["data"]["prepare_target"]["task_config"],
        multitask["trainer"]["task_config"],
    ]
    assert task_blocks[0] == task_blocks[1]
    for task in TASKS:
        single = read_config(CONFIG_ROOT / "experts" / "train" / f"{task}_sleepwalker.yml")
        assert task_blocks[0][task]["class_counts"] == single["trainer"]["class_counts"]


def test_main_configs_share_data_targets_and_training_options():
    configs = [read_config(CONFIG_ROOT / "main" / f"{method}.yml") for method in ("probability", "latent", "sei")]
    reference = configs[0]["trainer"]["task_config"]
    for method, config in zip(("probability", "latent", "sei"), configs):
        assert config["run"]["patients_per_epoch"] == 512
        assert config["run"]["package_path"] == f"iclr2026/models/{method}"
        assert "patient_group_size" not in config["run"]
        assert config["data"]["prepare_patient"]["keep_events"] == ["wake", "n1", "n2", "n3", "rem"]
        assert config["data"]["name"] == "sleepwalker.models.ModelGraphClassifier.paired_dataset_from_packages"
        assert config["data"]["dataset_class"] == "sleepwalker.datasets.HSP.HSP"
        assert set(config["data"]["packages"]) == set(TASKS)
        assert "channels" not in config["data"]
        assert "sample_frequency" not in config["data"]
        assert config["trainer"]["task_config"] == reference
        assert all(task["class_counts"] for task in config["trainer"]["task_config"].values())


def test_test_configs_inherit_channels_and_preprocessing_from_packages():
    for stem in {f"{task}_{model}" for task in TASKS for model in MODELS} | {"multitask_sleepwalker"}:
        test = read_config(CONFIG_ROOT / "experts" / "test" / f"{stem}.yml")
        assert set(test["data"]["dataset"]) == {"name", "event_mapping", "remove_unmapped_events"}


def test_sleepwalker_experts_use_portable_channel_sets():
    expected = {
        "arousal": ["EEG", "EOG", "Chin EMG", "ECG"],
        "breathing": ["Abdomen", "Chest", "Airflow", "SpO2"],
        "desaturation": ["SpO2"],
        "sleep": ["eeg"],
    }
    for task, logical_names in expected.items():
        config = read_config(CONFIG_ROOT / "experts" / "train" / f"{task}_sleepwalker.yml")
        assert [channel["logical_name"] for channel in config["data"]["channels"]] == logical_names


def test_osf_contract_is_consistent_across_tasks_and_test_configs():
    reference = read_config(CONFIG_ROOT / "experts" / "train" / "sleep_osf.yml")["data"]["channels"]
    assert len(reference) == 12
    for task in TASKS:
        train = read_config(CONFIG_ROOT / "experts" / "train" / f"{task}_osf.yml")
        test = read_config(CONFIG_ROOT / "experts" / "test" / f"{task}_osf.yml")
        assert train["data"]["channels"] == reference
        assert train["data"]["z_normalize"] is True
        assert "z_normalize" not in test["data"]["dataset"]


def test_sleepfm_contract_is_consistent_across_tasks_and_test_configs():
    reference = read_config(CONFIG_ROOT / "experts" / "train" / "sleep_sleepfm.yml")["data"]["channels"]
    assert len(reference) == 13
    assert [channel["logical_name"] for channel in reference] == ["ECG", "EMG_Chin", "EMG_LLeg", "EMG_RLeg", "ABD", "THX", "NP", "SpO2", "SN", "EOG_E1_A2", "EOG_E2_A1", "EEG_C3_A2", "EEG_C4_A1"]
    for task in TASKS:
        train = read_config(CONFIG_ROOT / "experts" / "train" / f"{task}_sleepfm.yml")
        test = read_config(CONFIG_ROOT / "experts" / "test" / f"{task}_sleepfm.yml")
        assert train["data"]["channels"] == reference
        assert train["data"]["sample_frequency"] == 128
        assert train["data"]["resample_type"] == "polyphase"
        assert train["data"]["total_input"] == "300s"
        assert train["data"]["z_normalize"] is True
        assert "z_normalize" not in test["data"]["dataset"]


def test_split_filter_covers_every_expert_channel_contract():
    split = read_config(CONFIG_ROOT / "split.yml")
    edf_filter = split["patient_filter"][0]
    osf = read_config(CONFIG_ROOT / "experts" / "train" / "sleep_osf.yml")
    osf_channels = {channel["logical_name"]: channel["physical_names"] for channel in osf["data"]["channels"]}
    sleepfm = read_config(CONFIG_ROOT / "experts" / "train" / "sleep_sleepfm.yml")
    sleepfm_channels = {channel["logical_name"]: channel["physical_names"] for channel in sleepfm["data"]["channels"]}
    desaturation = read_config(CONFIG_ROOT / "experts" / "train" / "desaturation_sleepwalker.yml")
    spo2 = next(channel for channel in desaturation["data"]["channels"] if channel["logical_name"] == "SpO2")

    assert edf_filter["name"] == "tools.split.filter_edf_files"
    assert {name: edf_filter["channels"][name] for name in osf_channels} == osf_channels
    assert {name: edf_filter["channels"][name] for name in sleepfm_channels} == sleepfm_channels
    assert edf_filter["channels"]["SpO2"] == spo2["physical_names"]


def test_runtime_configs_reuse_split_eligibility_without_refiltering():
    paths = list((CONFIG_ROOT / "experts").glob("*/*.yml")) + list((CONFIG_ROOT / "main").glob("*.yml"))
    for path in paths:
        assert "patient_filter" not in read_config(path), path
