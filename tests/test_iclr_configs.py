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
    assert {path.name for path in (CONFIG_ROOT / "main" / "train").glob("*.yml")} == {"end_to_end_sleepwalker.yml"}
    assert {path.name for path in (CONFIG_ROOT / "main" / "eval").glob("*.yml")} == {
        "end_to_end_sleepwalker.yml",
        "independent_osf.yml",
        "independent_sleepfm.yml",
        "independent_sleepwalker.yml",
    }
    assert {path.name for path in (CONFIG_ROOT / "main" / "train").iterdir() if path.is_dir()} == {"stacking"}
    assert {path.name for path in (CONFIG_ROOT / "main" / "eval").iterdir() if path.is_dir()} == {"stacking"}
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


def test_next_stacking_and_end_to_end_configs_are_matched():
    stacking_train = CONFIG_ROOT / "main" / "train" / "stacking"
    stacking_eval = CONFIG_ROOT / "main" / "eval" / "stacking"
    for family in ("sleepwalker", "sleepfm"):
        for regime in ("frozen", "task_local", "joint"):
            train = read_config(stacking_train / f"{regime}_{family}.yml")
            evaluation = read_config(stacking_eval / f"{regime}_{family}.yml")
            assert train["trainer"]["epochs"] == 2
            assert train["model"]["regime"] == regime.replace("_", "-")
            assert evaluation["system"]["package"] == train["run"]["package_path"]
            assert evaluation["test"]["prediction_output"].endswith(".feather")

    end_to_end = read_config(CONFIG_ROOT / "main" / "train" / "end_to_end_sleepwalker.yml")
    end_to_end_eval = read_config(CONFIG_ROOT / "main" / "eval" / "end_to_end_sleepwalker.yml")
    assert end_to_end["model"]["name"] == "sleepwalker.models.MultiTaskClassifierModel.MultiTaskClassifierModel"
    assert end_to_end["trainer"]["epochs"] == 100
    assert end_to_end["trainer"]["eval_every"] == 0
    assert end_to_end["run"]["test_repeats"] == []
    assert end_to_end_eval["system"]["package"] == end_to_end["run"]["package_path"]


def test_validation_protocol_configs_use_the_adaptation_split_and_full_test():
    stacking_train = CONFIG_ROOT / "main" / "train" / "stacking"
    stacking_eval = CONFIG_ROOT / "main" / "eval" / "stacking"
    pairs = {
        "frozen_sleepfm_validation": "frozen_sleepfm_validation",
        "exchange_sleep_sleepwalker_in_sleepfm_validation": "exchange_sleep_sleepwalker_in_sleepfm_validation",
        "exchange_breathing_sleepwalker_in_sleepfm_validation": "exchange_breathing_sleepwalker_in_sleepfm_validation",
        "joint_sleepfm_validation": "joint_sleepfm_validation",
        "rate_only_sleepfm_validation": "rate_only_sleepfm_validation",
        "frozen_sleepwalker_validation": "frozen_sleepwalker_validation",
        "joint_sleepwalker_validation": "joint_sleepwalker_validation",
        "frozen_osf_validation": "frozen_osf_validation",
        "joint_osf_validation": "joint_osf_validation",
        "rate_only_osf_validation": "rate_only_osf_validation",
        "exchange_sleep_sleepwalker_in_osf_validation": "exchange_sleep_sleepwalker_in_osf_validation",
        "exchange_breathing_sleepwalker_in_osf_validation": "exchange_breathing_sleepwalker_in_osf_validation",
    }
    for train_stem, eval_stem in pairs.items():
        train = read_config(stacking_train / f"{train_stem}.yml")
        evaluation = read_config(stacking_eval / f"{eval_stem}.yml")
        assert train["data"]["files"] == "results/iclr2026/main/adaptation_split.yml"
        assert train["trainer"]["epochs"] == 10
        assert train["run"]["tags"]["adaptation_split"] == "validation"
        assert evaluation["system"]["package"] == train["run"]["package_path"]
        assert evaluation["data"]["split"]["partition"] == "test"

    derived = (
        "aligned_independent_sleepwalker_validation",
        "aligned_independent_sleepfm_validation",
        "zero_shot_sw_sleep_in_sleepfm_validation",
        "zero_shot_sw_breathing_in_sleepfm_validation",
        "rate_only_zero_shot_sw_sleep_in_sleepfm_validation",
        "rate_only_zero_shot_sw_breathing_in_sleepfm_validation",
        "aligned_independent_osf_validation",
        "zero_shot_sw_sleep_in_osf_validation",
        "zero_shot_sw_breathing_in_osf_validation",
        "rate_only_zero_shot_sw_sleep_in_osf_validation",
        "rate_only_zero_shot_sw_breathing_in_osf_validation",
    )
    for stem in derived:
        evaluation = read_config(stacking_eval / f"{stem}.yml")
        assert evaluation["data"]["split"]["partition"] == "test"
        assert evaluation["test"]["prediction_output"].endswith(".feather")

    assert read_config(stacking_train / "rate_only_sleepfm_validation.yml")["model"]["alignment"] == "rate-only"
    assert read_config(stacking_train / "rate_only_osf_validation.yml")["model"]["alignment"] == "rate-only"


def test_trained_naive_zero_shot_configs_keep_the_host_evaluation_protocol():
    stacking_eval = CONFIG_ROOT / "main" / "eval" / "stacking"
    for family in ("sleepfm", "osf"):
        host = read_config(stacking_eval / f"naive_trained_{family}_validation.yml")
        for task in ("sleep", "breathing"):
            stem = f"naive_trained_zero_shot_sw_{task}_in_{family}_validation"
            swap = read_config(stacking_eval / f"{stem}.yml")
            assert swap["seed"] == host["seed"]
            assert swap["pipeline"] == host["pipeline"]
            assert swap["dependencies"] == host["dependencies"]
            assert swap["data"] == host["data"]
            assert swap["test"]["batch_size"] == host["test"]["batch_size"]
            assert swap["test"]["num_workers_dataloader"] == host["test"]["num_workers_dataloader"]
            assert swap["system"]["package"] == f"iclr2026/models/stacking/naive-trained-zero-shot-sw-{task}-in-{family}-validation"
            assert swap["test"]["output"].endswith(f"{stem}.jsonl")
            assert swap["test"]["prediction_output"].endswith(f"{stem}.feather")


def test_trained_naive_adaptation_matches_contract_refit_protocol():
    stacking_train = CONFIG_ROOT / "main" / "train" / "stacking"
    stacking_eval = CONFIG_ROOT / "main" / "eval" / "stacking"
    for family in ("sleepfm", "osf"):
        for task in ("sleep", "breathing"):
            stem = f"exchange_{task}_sleepwalker_in_{family}_validation"
            naive_stem = f"naive_trained_{stem}"
            contract_train = read_config(stacking_train / f"{stem}.yml")
            naive_train = read_config(stacking_train / f"{naive_stem}.yml")
            contract_eval = read_config(stacking_eval / f"{stem}.yml")
            naive_eval = read_config(stacking_eval / f"{naive_stem}.yml")

            assert naive_train["data"] == contract_train["data"]
            assert naive_train["trainer"] == contract_train["trainer"]
            assert naive_train["model"]["alignment"] == "naive"
            assert {key: value for key, value in naive_train["model"].items() if key != "alignment"} == contract_train["model"]
            assert naive_train["run"]["batch_size"] == contract_train["run"]["batch_size"]
            assert naive_train["run"]["n_samples"] == contract_train["run"]["n_samples"]
            assert naive_train["run"]["package_path"] == naive_eval["system"]["package"]
            assert naive_eval["pipeline"] == contract_eval["pipeline"]
            assert naive_eval["dependencies"] == contract_eval["dependencies"]
            assert naive_eval["data"] == contract_eval["data"]
            assert naive_eval["test"]["batch_size"] == contract_eval["test"]["batch_size"]
            assert naive_eval["test"]["num_workers_dataloader"] == contract_eval["test"]["num_workers_dataloader"]
            assert naive_eval["test"]["output"].endswith(f"{naive_stem}.jsonl")
            assert naive_eval["test"]["prediction_output"].endswith(f"{naive_stem}.feather")


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
