from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from iclr2026.scripts.replace_yaml_block import materialize_configs


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_run_script():
    spec = importlib.util.spec_from_file_location("iclr2026_run", REPO_ROOT / "iclr2026" / "scripts" / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shell_expansions_cover_all_expert_configs():
    train_configs = sorted((REPO_ROOT / "iclr2026" / "configs" / "experts" / "train").glob("*.yml"))
    test_configs = sorted((REPO_ROOT / "iclr2026" / "configs" / "experts" / "test").glob("*.yml"))

    assert len(train_configs) == 12
    assert len(test_configs) == 12


def test_run_script_orders_required_tables_external_work_and_seeds():
    script_path = REPO_ROOT / "iclr2026" / "scripts" / "run.sh"
    script = script_path.read_text(encoding="utf-8")

    assert script_path.stat().st_mode & 0o111
    assert "job shhs-sw evaluate" in script
    assert script.index("job shhs-table summary") < script.index("job train-probability-sw train") < script.index("job required-complete barrier") < script.index('job "train-probability-sw-seed$seed"')
    assert 'GPUS=${SLEEPWALKER_ICLR2026_GPUS:-0,1,2}' in script
    assert not (REPO_ROOT / "iclr2026" / "scripts" / "prepare.sh").exists()
    assert "multitask_sleepwalker" not in script


def test_jobs_are_small_explicit_records_and_barrier_depends_on_previous_jobs():
    run_script = load_run_script()

    jobs = run_script.parse_jobs(["train|train|train.yml|", "eval|system|eval.yml|train", "done|barrier|done|"])

    assert [(job.name, job.kind, job.dependencies) for job in jobs] == [("train", "train", []), ("eval", "system", ["train"]), ("done", "barrier", ["train", "eval"])]


def test_completed_training_and_evaluation_jobs_are_recognized(tmp_path, monkeypatch):
    run_script = load_run_script()
    monkeypatch.chdir(tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text("{}", encoding="utf-8")
    (package / "model.pt").write_bytes(b"model")
    train = tmp_path / "train.yml"
    train.write_text(yaml.safe_dump({"run": {"package_path": str(package)}}), encoding="utf-8")
    output = tmp_path / "output.jsonl"
    output.write_text('{"record_type":"run"}\n{"record_type":"aggregate"}\n', encoding="utf-8")
    evaluation = tmp_path / "evaluation.yml"
    evaluation.write_text(yaml.safe_dump({"test": {"output": str(output)}}), encoding="utf-8")

    assert run_script.job_is_complete(run_script.Job("train", "train", str(train), []))
    assert run_script.job_is_complete(run_script.Job("eval", "evaluate", str(evaluation), []))


def test_gpu_and_session_names_are_validated():
    run_script = load_run_script()

    assert run_script.parse_gpus("0, 1,2") == ["0", "1", "2"]
    assert run_script.session_name("paper run", "sleep-sleepwalker") == "iclr26-paper-run-sleep-sleepwalker"
    with pytest.raises(ValueError, match="without duplicates"):
        run_script.parse_gpus("0,0")


def test_queue_waits_until_tmux_reports_the_dead_pane_status(monkeypatch):
    run_script = load_run_script()
    responses = [SimpleNamespace(returncode=0), SimpleNamespace(returncode=0, stdout="1\t\n")]
    monkeypatch.setattr(run_script.subprocess, "run", lambda *args, **kwargs: responses.pop(0))

    assert run_script.pane_state("session") == (True, None)


def test_job_commands_use_the_requested_evaluator_and_training_mode(tmp_path):
    run_script = load_run_script()
    python = tmp_path / "python"

    assert run_script.job_command(run_script.Job("train", "train", "train.yml", []), python) == [str(python), "tools/train.py", "train", "train.yml"]
    assert run_script.job_command(run_script.Job("external", "evaluate", "test.yml", []), python) == [str(python), "tools/evaluate.py", "--overwrite", "test.yml"]
    assert run_script.job_command(run_script.Job("paper", "system", "system.yml", []), python) == [str(python), "tools/evaluate_system.py", "--overwrite", "system.yml"]


def test_replace_yaml_block_materializes_complete_configs(tmp_path):
    sources = []
    for name in ("probability", "sei"):
        source = tmp_path / "source" / f"{name}.yml"
        source.parent.mkdir(exist_ok=True)
        source.write_text(yaml.safe_dump({"seed": 17, "system": {"name": name}, "data": {"label": "HSP"}, "test": {"batch_size": 64, "output": f"results/hsp/{name}.jsonl"}}, sort_keys=False), encoding="utf-8")
        sources.append(source)
    replacement = tmp_path / "ruhrland.yml"
    replacement.write_text(yaml.safe_dump({"label": "Ruhrland", "dataset": {"name": "sleepwalker.datasets.Ruhrlandklinik.Ruhrlandklinik"}}, sort_keys=False), encoding="utf-8")

    outputs = materialize_configs(sources=sources, block="data", replacement=replacement, output_config_dir=tmp_path / "generated", output_results_dir=Path("results/external/ruhrland"))

    assert [path.name for path in outputs] == ["probability.yml", "sei.yml"]
    probability = yaml.safe_load(outputs[0].read_text(encoding="utf-8"))
    assert probability["data"] == yaml.safe_load(replacement.read_text(encoding="utf-8"))
    assert probability["system"] == {"name": "probability"}
    assert probability["test"] == {"batch_size": 64, "output": "results/external/ruhrland/probability.jsonl"}


def test_replace_yaml_block_refuses_existing_outputs(tmp_path):
    source = tmp_path / "source.yml"
    replacement = tmp_path / "replacement.yml"
    destination = tmp_path / "generated"
    source.write_text(yaml.safe_dump({"data": {"label": "HSP"}, "test": {"output": "result.jsonl"}}), encoding="utf-8")
    replacement.write_text(yaml.safe_dump({"label": "SHHS"}), encoding="utf-8")
    materialize_configs(sources=[source], block="data", replacement=replacement, output_config_dir=destination, output_results_dir=tmp_path / "results")

    with pytest.raises(FileExistsError, match="already exist"):
        materialize_configs(sources=[source], block="data", replacement=replacement, output_config_dir=destination, output_results_dir=tmp_path / "results")


def test_external_configs_are_synchronized_with_iclr_sources():
    shhs_dir = REPO_ROOT / "configs" / "other" / "shhs"
    shhs_sources = [REPO_ROOT / "iclr2026" / "configs" / "experts" / "test" / f"sleep_{family}.yml" for family in ("sleepwalker", "sleepfm", "osf")]
    shhs_data = yaml.safe_load((shhs_dir / "data.yml").read_text(encoding="utf-8"))
    expected_channels = {
        "sleep_sleepwalker.yml": ["eeg"],
        "sleep_sleepfm.yml": ["ECG", "EMG_Chin", "EMG_LLeg", "EMG_RLeg", "ABD", "THX", "NP", "SpO2", "SN", "EOG_E1_A2", "EOG_E2_A1", "EEG_C3_A2", "EEG_C4_A1"],
        "sleep_osf.yml": ["ECG", "EMG_Chin", "EMG_LLeg", "EMG_RLeg", "ABD", "THX", "NP", "SN", "EOG_E1_A2", "EOG_E2_A1", "EEG_C3_A2", "EEG_C4_A1"],
    }

    assert {path.name for path in shhs_dir.glob("*.yml") if path.name != "data.yml"} == {path.name for path in shhs_sources}
    for source in shhs_sources:
        generated = yaml.safe_load((shhs_dir / source.name).read_text(encoding="utf-8"))
        assert generated["data"]["label"] == shhs_data["label"]
        assert generated["data"]["split"] == shhs_data["split"]
        assert generated["data"]["dataset"]["event_mapping"] == shhs_data["dataset"]["event_mapping"]
        assert generated["data"]["dataset"]["group_sampling_strategy"] == "first"
        assert [channel["logical_name"] for channel in generated["data"]["dataset"]["channels"]] == expected_channels[source.name]
        channels = {channel["logical_name"]: channel["physical_names"] for channel in generated["data"]["dataset"]["channels"]}
        if source.name != "sleep_sleepwalker.yml":
            assert channels["EEG_C3_A2"] == ["EEG(sec)", "EEG2"]
            assert channels["NP"] == ["NEW AIR", "AIRFLOW", "THOR RES"]
        assert generated["test"]["output"] == f"results/iclr2026/other/shhs/{source.name.removesuffix('.yml')}.jsonl"

    expert_sources = sorted((REPO_ROOT / "iclr2026" / "configs" / "experts" / "test").glob("*.yml"))
    graph_sources = sorted((REPO_ROOT / "iclr2026" / "configs" / "external" / "graphs").glob("*.yml"))
    ruhrland_sources = expert_sources + graph_sources
    assert len(ruhrland_sources) == 21
    for cohort in ("pre2024", "2024"):
        cohort_dir = REPO_ROOT / "configs" / "other" / "ruhrland" / cohort
        cohort_data = yaml.safe_load((cohort_dir / "data.yml").read_text(encoding="utf-8"))
        sleepfm_data = yaml.safe_load((REPO_ROOT / "configs" / "other" / "ruhrland" / f"foundation-{cohort}.yml").read_text(encoding="utf-8"))
        osf_data = yaml.safe_load((REPO_ROOT / "configs" / "other" / "ruhrland" / f"foundation-osf-{cohort}.yml").read_text(encoding="utf-8"))
        assert {path.name for path in cohort_dir.glob("*.yml") if path.name != "data.yml"} == {path.name for path in ruhrland_sources}
        for source in ruhrland_sources:
            expected = yaml.safe_load(source.read_text(encoding="utf-8"))
            expected["data"] = sleepfm_data if source.stem.endswith("sleepfm") else osf_data if source.stem.endswith("osf") else cohort_data
            expected["test"]["output"] = f"results/iclr2026/other/ruhrland/{cohort}/{source.stem}.jsonl"
            generated = yaml.safe_load((cohort_dir / source.name).read_text(encoding="utf-8"))
            assert generated == expected


def test_external_data_blocks_cover_required_labels_and_distinct_cohorts():
    shhs = yaml.safe_load((REPO_ROOT / "configs" / "other" / "shhs" / "data.yml").read_text(encoding="utf-8"))
    pre2024 = yaml.safe_load((REPO_ROOT / "configs" / "other" / "ruhrland" / "pre2024" / "data.yml").read_text(encoding="utf-8"))
    ruhrland2024 = yaml.safe_load((REPO_ROOT / "configs" / "other" / "ruhrland" / "2024" / "data.yml").read_text(encoding="utf-8"))

    assert shhs["dataset"]["name"] == "sleepwalker.datasets.SHHS.SHHS"
    assert shhs["dataset"]["annotator"] == "nsrr"
    assert set(shhs["dataset"]["event_mapping"].values()) == {"wake", "n1", "n2", "n3", "rem"}
    assert pre2024["dataset"] == ruhrland2024["dataset"]
    assert set(pre2024["dataset"]["event_mapping"].values()) == {"wake", "n1", "n2", "n3", "rem", "arousal", "apnea", "hypopnea", "desaturation"}
    assert pre2024["label"] == "Ruhrland-pre-2024"
    assert ruhrland2024["label"] == "Ruhrland-2024"
    assert pre2024["split"]["file"] != ruhrland2024["split"]["file"]

    for cohort in ("pre2024", "2024"):
        sleepfm = yaml.safe_load((REPO_ROOT / "configs" / "other" / "ruhrland" / f"foundation-{cohort}.yml").read_text(encoding="utf-8"))
        osf = yaml.safe_load((REPO_ROOT / "configs" / "other" / "ruhrland" / f"foundation-osf-{cohort}.yml").read_text(encoding="utf-8"))
        sleepfm_channels = [channel["logical_name"] for channel in sleepfm["dataset"]["channels"]]
        osf_channels = [channel["logical_name"] for channel in osf["dataset"]["channels"]]
        assert sleepfm_channels == osf_channels[:7] + ["SpO2"] + osf_channels[7:]
        assert sleepfm["dataset"]["group_sampling_strategy"] == "first"
        assert osf["dataset"]["group_sampling_strategy"] == "first"


def test_external_runner_uses_general_package_evaluation_and_covers_generated_configs():
    script_path = REPO_ROOT / "iclr2026" / "scripts" / "run.sh"
    script = script_path.read_text(encoding="utf-8")
    shhs_names = {path.name for path in (REPO_ROOT / "configs" / "other" / "shhs").glob("*.yml") if path.name != "data.yml"}
    ruhrland_names = {path.name for path in (REPO_ROOT / "configs" / "other" / "ruhrland" / "pre2024").glob("*.yml") if path.name != "data.yml"}

    assert script_path.stat().st_mode & 0o111
    assert "job shhs-sw evaluate configs/other/shhs/sleep_sleepwalker.yml" in script
    assert script.count("--fractions 0 0 1") == 3
    for name in shhs_names:
        assert f"configs/other/shhs/{name}" in script
    assert len(ruhrland_names) == 21
    assert "Optional collaborator sweep: graph packages evaluated directly with tools/evaluate.py." in script
