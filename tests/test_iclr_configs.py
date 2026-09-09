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
    expected = {f"{task}_{model}" for task in TASKS for model in MODELS}

    assert {path.stem for path in train} == expected
    assert {path.stem for path in test} == expected
    assert {path.name for path in (CONFIG_ROOT / "main" / "train").glob("*.yml")} == {
        "probability_sleepwalker.yml", "latent_sleepwalker.yml", "refit_heads_sleepwalker.yml", "sei_k8_sleepwalker.yml", "sei_k32_sleepwalker.yml", "sei_k8_sleepfm.yml", "sei_k8_osf.yml",
        "probability_sleepwalker_seed23.yml", "probability_sleepwalker_seed42.yml", "sei_k8_sleepwalker_seed23.yml", "sei_k8_sleepwalker_seed42.yml",
    }
    assert len(list((CONFIG_ROOT / "main" / "eval").glob("*.yml"))) == 26
    assert not (CONFIG_ROOT / "generate_configs.py").exists()
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

        prepare_target = config["data"]["prepare_target"]
        assert prepare_target["target_resolution"]
        assert prepare_target["target_resolution"] == config["trainer"]["target_resolution"]
        assert prepare_target["target_offset"] == config["trainer"]["target_offset"]
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
    assert len(package_paths) == len(TASKS) * len(MODELS)


def test_task_class_counts_match_across_expert_variants():
    for task in TASKS:
        sleepwalker = read_config(CONFIG_ROOT / "experts" / "train" / f"{task}_sleepwalker.yml")
        for foundation in ("osf", "sleepfm"):
            probe = read_config(CONFIG_ROOT / "experts" / "train" / f"{task}_{foundation}.yml")
            assert sleepwalker["trainer"]["class_counts"] == probe["trainer"]["class_counts"]


def test_osf_experts_use_one_epoch_level_embedding():
    for task in ("arousal", "breathing", "desaturation"):
        config = read_config(CONFIG_ROOT / "experts" / "train" / f"{task}_osf.yml")
        assert config["data"]["total_input"] == "30s"
        assert config["data"]["stride"] == "30s"
        assert config["data"]["prepare_target"]["target_resolution"] == "30s"
        assert config["data"]["prepare_target"]["target_offset"] == "0s"
        assert config["data"]["prepare_target"]["sequence_len"] == 1
        assert config["trainer"]["target_resolution"] == "30s"
        assert config["trainer"]["target_offset"] == "0s"
        assert config["trainer"]["sequence_len"] == 1
        assert config["model"]["head"] == "linear"
        assert config["model"]["freeze_encoder"] is True


def test_graphs_use_explicit_family_independent_output_sequences():
    expected = {
        "sleep": ("30s", 1, "25s"),
        "arousal": ("1s", 40, "20s"),
        "breathing": ("5s", 8, "20s"),
        "desaturation": ("10s", 8, "0s"),
    }
    for path in (CONFIG_ROOT / "main" / "train").glob("*.yml"):
        config = read_config(path)
        task_config = config["trainer"]["task_config"]
        assert {task: (values["target_resolution"], values["sequence_len"], values["target_offset"]) for task, values in task_config.items()} == expected
        assert config["data"]["total_input"] == "80s"
        assert config["data"]["stride"] == "30s"
        assert "graph_outputs" not in config["data"]
        assert config["model"]["nodes"]["graph_outputs"] == task_config
        assert "graph_total_input" not in config["model"]["nodes"]

def test_main_configs_share_data_targets_and_training_options():
    names = ("probability_sleepwalker", "latent_sleepwalker", "sei_k8_sleepwalker", "sei_k32_sleepwalker", "refit_heads_sleepwalker")
    configs = [read_config(CONFIG_ROOT / "main" / "train" / f"{name}.yml") for name in names]
    reference = configs[0]["trainer"]["task_config"]
    for config in configs:
        assert config["run"]["patients_per_epoch"] == 512
        assert config["run"]["package_path"].startswith("iclr2026/models/")
        assert config["run"]["test_repeats"] == []
        assert config["trainer"]["eval_every"] == 0
        assert config["trainer"]["return_best"] is False
        assert "patient_group_size" not in config["run"]
        assert config["data"]["prepare_patient"]["keep_events"] == ["wake", "n1", "n2", "n3", "rem"]
        assert config["data"]["name"] == "sleepwalker.datasets.HSP.HSP"
        assert config["data"]["channels"] == []
        assert config["data"]["sample_frequency"] == 1
        assert set(config["model"]["nodes"]["packages"]) == set(TASKS)
        assert "packages" not in config["data"]
        assert config["trainer"]["task_config"] == reference
        assert all(task["class_counts"] for task in config["trainer"]["task_config"].values())


def test_test_configs_inherit_channels_and_preprocessing_from_packages():
    for stem in {f"{task}_{model}" for task in TASKS for model in MODELS}:
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


def test_foundation_sleep_heads_use_their_native_temporal_outputs():
    osf = read_config(CONFIG_ROOT / "experts" / "train" / "sleep_osf.yml")
    assert osf["data"]["total_input"] == "30s"
    assert osf["data"]["stride"] == "30s"
    assert osf["trainer"]["target_resolution"] == "30s"
    assert osf["trainer"]["sequence_len"] == 1

    sleepfm = read_config(CONFIG_ROOT / "experts" / "train" / "sleep_sleepfm.yml")
    assert sleepfm["data"]["total_input"] == "300s"
    assert sleepfm["data"]["stride"] == "300s"
    assert sleepfm["trainer"]["target_resolution"] == "300s"
    assert sleepfm["data"]["prepare_target"]["sequence_len"] == 10
    assert sleepfm["trainer"]["sequence_len"] == 10
    assert sleepfm["model"]["name"] == "sleepwalker.models.PackagedSequenceClassifierModel.PackagedSequenceClassifierModel"
    assert sleepfm["model"]["token_count"] == 60


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
