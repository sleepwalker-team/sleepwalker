from __future__ import annotations

from types import SimpleNamespace

import pytest

import sleepwalker.cli.main as cli
import sleepwalker.cli.train as train_cli
import sleepwalker.cli.evaluate as evaluate_cli
import sleepwalker.cli.evaluate_system as system_cli


@pytest.mark.parametrize("command", cli.COMMANDS)
def test_external_id_is_logged_and_passed_explicitly_to_training(monkeypatch, command):
    observed = []
    contexts = []
    monkeypatch.setattr(cli.logger, "context", lambda value: contexts.append(value))
    monkeypatch.setattr(cli.logger, "uncontext", lambda: contexts.pop())

    def execute(arguments, *, run_id=None):
        observed.append((arguments, run_id, list(contexts)))
        return 7

    monkeypatch.setitem(cli.COMMANDS, command, execute)

    assert cli.main([command, "--id", "haml-id", "payload", "--existing-option"]) == 7
    assert observed == [(["payload", "--existing-option"], "haml-id" if command in {"train", "dry", "resume"} else None, ["haml-id"])]
    assert contexts == []


@pytest.mark.parametrize("command", ["train", "dry"])
@pytest.mark.parametrize("arguments", [
    ["--id", "haml-id", "COMMAND", "--config", "recipe.yml", "--fold", "fold_2"],
    ["COMMAND", "--config", "recipe.yml", "--id", "haml-id", "--fold", "fold_2"],
    ["COMMAND", "recipe.yml", "--fold", "fold_2", "--id=haml-id"],
])
def test_training_records_id_without_changing_recipe(monkeypatch, command, arguments):
    config = {"run": {"experiment_name": "original", "package_path": "models/expert"}, "seed": 17}
    cfg = SimpleNamespace(tags={"task": "sleep"}, meta_data={"seed": 17}, experiment_name="original", package_path="models/expert")
    calls = []
    monkeypatch.setattr(train_cli, "read_yaml", lambda path: calls.append(("read", path)) or config)
    monkeypatch.setattr(train_cli, "runcfg_from_dict", lambda recipe, dry_run, fold: calls.append(("build", recipe, dry_run, fold)) or cfg)
    monkeypatch.setattr(train_cli, "run", lambda run_config: calls.append(("run", run_config)))

    assert cli.main([command if argument == "COMMAND" else argument for argument in arguments]) == 0

    assert calls == [("read", "recipe.yml"), ("build", config, command == "dry", "fold_2"), ("run", cfg)]
    assert cfg.tags == {"task": "sleep", "run_id": "haml-id"}
    assert cfg.meta_data == {"seed": 17, "run_id": "haml-id"}
    assert (cfg.experiment_name, cfg.package_path) == ("original", "models/expert")
    assert config == {"run": {"experiment_name": "original", "package_path": "models/expert"}, "seed": 17}


def test_external_id_context_is_restored_after_failure(monkeypatch):
    contexts = []
    monkeypatch.setattr(cli.logger, "context", lambda value: contexts.append(value))
    monkeypatch.setattr(cli.logger, "uncontext", lambda: contexts.pop())

    def fail(arguments, *, run_id=None):
        assert run_id == "haml-id"
        raise SystemExit(5)

    monkeypatch.setitem(cli.COMMANDS, "train", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["train", "recipe.yml", "--id", "haml-id"])
    assert error.value.code == 5
    assert contexts == []


@pytest.mark.parametrize("arguments", [["train", "recipe.yml", "--id"], ["train", "recipe.yml", "--id", ""], ["train", "recipe.yml", "--id", "  "]])
def test_invalid_run_id_fails_before_dispatch(monkeypatch, arguments):
    dispatched = []
    monkeypatch.setitem(cli.COMMANDS, "train", lambda remaining: dispatched.append(remaining))
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
    assert dispatched == []


@pytest.mark.parametrize("command", ["train", "dry"])
@pytest.mark.parametrize("arguments", [[], ["recipe.yml", "--config", "other.yml"]])
def test_config_path_is_required_and_unambiguous(command, arguments):
    with pytest.raises(SystemExit) as error:
        train_cli.build_parser().parse_args([command, *arguments])
    assert error.value.code == 2


def test_checkpoint_resumption_records_launch_id(monkeypatch):
    cfg = SimpleNamespace(tags={}, meta_data={})
    calls = []
    monkeypatch.setattr(train_cli, "runcfg_from_checkpoint", lambda path: calls.append(("load", path)) or cfg)
    monkeypatch.setattr(train_cli, "run", lambda value: calls.append(("run", value)))

    cli.main(["resume", "checkpoint.pt", "--id", "resume-id"])

    assert calls == [("load", "checkpoint.pt"), ("run", cfg)]
    assert cfg.tags["run_id"] == cfg.meta_data["run_id"] == "resume-id"


@pytest.mark.parametrize('command,module', [('evaluate', evaluate_cli), ('evaluate-system', system_cli)])
@pytest.mark.parametrize('arguments', [['recipe.yml'], ['--config', 'recipe.yml']])
def test_evaluation_accepts_haml_configs(monkeypatch, command, module, arguments):
    config = {'test': {'package': 'models/expert'}}
    calls = []
    monkeypatch.setattr(module, 'read_config', lambda path: calls.append(('read', path)) or config)
    monkeypatch.setattr(module, 'execute', lambda *values: calls.append(('execute', values)))

    cli.main([command, *arguments, '--id', 'eval-id'])

    assert calls[0] == ('read', 'recipe.yml')
    assert calls[1] == ('execute', ('models/expert', config) if command == 'evaluate' else (config, 'recipe.yml'))
