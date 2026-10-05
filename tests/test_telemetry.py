import json
import logging
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import torch
import yaml

import sleepwalker.cli.train as train_cli
import sleepwalker.telemetry as telemetry
import sleepwalker.trainer.Run as run_module
from sleepwalker.telemetry import TelemetrySink
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run, seed_everything
from sleepwalker.utils import LevelAwareFormatter, Sink, UnifiedLogger


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def make_logger(sink):
    logger = UnifiedLogger(LevelAwareFormatter("%(message)s", "%(message)s"), level=logging.INFO)
    logger.add_sink(sink)
    return logger


def batch(patients=("a", "b")):
    return {"data": torch.ones(2, 4, 1), "target": torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]]), "target_mask": torch.tensor([[True, False], [True, True]]), "patient": list(patients), "time": ["t1", "t2"]}


def test_batch_received_dispatches_read_only_payload_to_multiple_sinks():
    observations = []

    class Observer(Sink):
        def batch_received(self, batch, context):
            observations.append((batch, context))

    logger = make_logger(Observer())
    logger.add_sink(Observer())
    logger.start_run("run")
    logger.context("TRAIN [1/2]")
    item = batch()
    logger.batch_received(item)
    logger.uncontext()
    logger.end_run()
    assert len(observations) == 2
    assert all(payload is item for payload, context in observations)
    assert [context for payload, context in observations] == ["run | TRAIN [1/2]"] * 2


def test_resource_sink_records_duration_without_training_accounting(tmp_path):
    sink = TelemetrySink(tmp_path, resources=False)
    logger = make_logger(sink)
    logger.start_run("run")
    logger.batch_received(batch())
    logger.metric("epoch/train/loss", 0.25)
    logger.end_run()
    assert {path.name for path in tmp_path.iterdir()} == {"summary.json"}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "FINISHED"
    assert summary["elapsed_s"] >= 0


class TinyDataset:
    sample_frequency = 100
    total_input = "40ms"
    stride = "40ms"
    initialized = True

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"data": torch.full((4, 1), float(index)), "target": torch.nn.functional.one_hot(torch.tensor([index % 2]), 2).float(), "patient": f"patient-{index // 2}", "time": f"window-{index}"}

    def get_n_patients(self):
        return 2

    def get_classes(self):
        return ["wake", "rem"]

    def get_input_channels(self):
        return ["eeg"]

    def get_timeseries_len(self):
        return 4

    def set_rejection_strategy(self, strategy):
        assert strategy == "none"


class TinyModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.fail = fail

    def forward(self, data):
        if self.fail:
            raise RuntimeError("model failed")
        return self.linear(data.mean(dim=1)).unsqueeze(1)

    def input_spec(self):
        return (1, 4, 1), {"layout": "BTC"}

    def warmup_preprocessor(self, loader, device):
        return self


def make_run(path, *, enabled, fail=False):
    seed_everything(17)
    dataset = TinyDataset()
    trainer = MulticlassTrainer(epochs=1, optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.01), classes=dataset.get_classes(), loss_function=torch.nn.functional.cross_entropy, target_resolution="1s", device="cpu", warmup_device="cpu", save_every=0, eval_every=1, return_best=False)
    return RunCfg(experiment_name="tiny", model_name="tiny", model=TinyModel(fail), trainer=trainer, train_datasets=[dataset], val_datasets=[dataset], test_datasets=[("test", dataset)], batch_size=2, n_samples=4, num_workers_dataloader=0, collate_fn=torch.utils.data.default_collate, log_path=str(path), export_package=False, telemetry={"resources": False} if enabled else None)


def test_batch_observation_preserves_training_and_receives_stage_contexts(tmp_path):
    plain = make_run(tmp_path / "plain", enabled=False)
    measured = make_run(tmp_path / "measured", enabled=True)
    rng = torch.get_rng_state()
    run(plain)
    torch.set_rng_state(rng)
    observations = []

    class Observer(Sink):
        def batch_received(self, batch, context):
            assert batch["data"].device.type == "cpu"
            observations.append((len(batch["data"]), context))

    sink = Observer()
    run_module.logger.add_sink(sink)
    try:
        run(measured)
    finally:
        run_module.logger.remove_sink(sink)
    for name, value in plain.model.state_dict().items():
        assert torch.equal(value, measured.model.state_dict()[name])
    assert plain.trainer.losses == measured.trainer.losses
    assert observations == [(2, "tiny | TRAIN [1/1]")] * 2 + [(2, "tiny | VAL [1/1]")] * 2 + [(2, "tiny | test | TEST")] * 2


@pytest.mark.parametrize("failure", ["model", "warmup", "sink"])
def test_run_failure_closes_resources_and_removes_its_sinks(tmp_path, monkeypatch, failure):
    cfg = make_run(tmp_path, enabled=True, fail=failure == "model")
    # Reach the real trainer loop rather than failing during torchinfo's forward.
    monkeypatch.setattr(run_module, "summary", lambda *args, **kwargs: None)
    if failure == "warmup":
        def fail_warmup(loader):
            raise RuntimeError("warmup failed")
        monkeypatch.setattr(cfg.trainer, "warmup_trainer", fail_warmup)
    elif failure == "sink":
        def fail_batch(batch):
            raise RuntimeError("sink failed")
        monkeypatch.setattr(run_module.logger, "batch_received", fail_batch)
    sinks = list(run_module.logger._sinks)
    context = run_module.logger._ctx_str()
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        run(cfg)
    assert run_module.logger._sinks == sinks
    assert run_module.logger._ctx_str() == context
    output = tmp_path / "tiny/telemetry"
    assert json.loads((output / "summary.json").read_text())["status"] == "FAILED"


def test_resource_sampling_runs_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "inventory", lambda: {"test": True})
    sink = TelemetrySink(tmp_path, interval=0.01)
    sampled = threading.Event()
    original_snapshot = sink.snapshot
    samples = []

    def snapshot():
        sample = original_snapshot()
        samples.append(sample)
        if len(samples) >= 2:
            sampled.set()
        return sample

    monkeypatch.setattr(sink, "snapshot", snapshot)
    logger = make_logger(sink)
    logger.start_run("resources")
    # Wait on sampling itself, without an arbitrary sleep or GPU work.
    assert sampled.wait(2)
    logger.end_run()
    records = read_records(tmp_path / "telemetry.jsonl")
    assert len(records) >= 2
    assert "process_tree_cpu_core_pct" in records[-1]
    assert not sink.thread.is_alive()
    assert all(handle.closed for handle in sink.handles.values())


def test_resource_errors_are_reported_and_shutdown_still_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "inventory", lambda: {})
    sink = TelemetrySink(tmp_path)

    def fail_snapshot():
        raise OSError("probe failed")

    monkeypatch.setattr(sink, "snapshot", fail_snapshot)
    logger = make_logger(sink)
    logger.start_run("bad-resources")
    sink.thread.join(timeout=2)
    with pytest.raises(RuntimeError, match="recording failed"):
        logger.end_run()
    assert not sink.thread.is_alive()
    assert all(handle.closed for handle in sink.handles.values())


def test_logger_stops_every_sink_when_one_end_fails(tmp_path):
    closed = []

    class FailingSink(Sink):
        def end(self, status):
            closed.append("first")
            raise RuntimeError("end failed")

    class OtherSink(Sink):
        def end(self, status):
            closed.append("second")

    logger = make_logger(FailingSink())
    logger.add_sink(OtherSink())
    logger.start_run("cleanup")
    with pytest.raises(RuntimeError, match="end failed"):
        logger.end_run()
    assert closed == ["first", "second"]
    assert logger._ctx_str() == ""


def test_existing_telemetry_is_not_overwritten(tmp_path):
    (tmp_path / "summary.json").write_text("previous\n")
    sink = TelemetrySink(tmp_path, resources=False)
    logger = make_logger(sink)
    with pytest.raises(FileExistsError):
        logger.start_run("duplicate")
    assert (tmp_path / "summary.json").read_text() == "previous\n"


def test_checkpoint_resume_preserves_previous_measurements(tmp_path, monkeypatch):
    cfg = make_run(tmp_path, enabled=True)
    previous = tmp_path / "telemetry"
    previous.mkdir()
    (previous / "summary.json").write_text("previous measurements\n")
    config = {"data": {}, "model": {"name": "tiny"}, "trainer": {}, "run": {"experiment_name": "tiny", "model_name": "tiny", "batch_size": 2, "n_samples": 4, "num_workers_dataloader": 0, "telemetry": {"output": str(previous), "resources": False}}}
    checkpoint = tmp_path / "final/checkpoint.pt"
    checkpoint.parent.mkdir()
    (tmp_path / "hparams.yml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(train_cli.BaseTrainer, "load_checkpoint", lambda path: (cfg.trainer, cfg.model))
    monkeypatch.setattr(train_cli, "initialize_datasets", lambda *args, **kwargs: (cfg.train_datasets, cfg.val_datasets, cfg.test_datasets))

    restored = train_cli.runcfg_from_checkpoint(checkpoint)

    directory = Path(restored.telemetry["output"])
    assert directory.parent == previous
    assert directory.name.startswith("resume-")
    assert restored.meta_data["run"]["telemetry"]["output"] == str(directory)
    assert (previous / "summary.json").read_text() == "previous measurements\n"
    assert yaml.safe_load((tmp_path / "hparams.yml").read_text()) == config


def test_resource_rates_keep_device_counters_separate_and_ignore_reused_pids(tmp_path, monkeypatch):
    sink = TelemetrySink(tmp_path, resources=False)
    monkeypatch.setattr(telemetry.os, "sysconf", lambda name: 100)
    sink.previous = {"elapsed_s": 0., "host_cpu_ticks": [100, 0, 50, 200, 0, 0, 0, 0], "processes": {42: {"ticks": 10, "start_ticks": 1}, 43: {"ticks": 100, "start_ticks": 2}}, "disks": {"nvme0n1": {"read_bytes": 2000, "write_bytes": 500, "busy_ms": 300}}}
    sample = {"elapsed_s": 2., "host_cpu_ticks": [110, 0, 60, 220, 10, 0, 0, 0], "processes": {42: {"ticks": 50, "start_ticks": 1, "rss_bytes": 1000}, 43: {"ticks": 999, "start_ticks": 3, "rss_bytes": 2000}}, "disks": {"nvme0n1": {"read_bytes": 6000, "write_bytes": 1500, "busy_ms": 800}}}

    sink.rates(sample)

    assert sample["host_cpu_utilization_pct"] == 40.
    assert sample["host_iowait_pct"] == 20.
    assert sample["process_tree_cpu_core_pct"] == 20.
    assert sample["process_tree_rss_sum_bytes"] == 3000
    assert sample["disks"]["nvme0n1"]["read_bytes_s"] == 2000.
    assert sample["disks"]["nvme0n1"]["busy_pct"] == 25.


def test_gpu_probe_uses_the_selected_physical_identifier(monkeypatch):
    commands = []
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def execute(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="GPU-abc, NVIDIA A30, 72, 11, 100, 24576\n")

    monkeypatch.setattr(telemetry.subprocess, "run", execute)
    result = telemetry.gpu_stats("GPU-abc")
    assert commands[0][1:3] == ["-i", "GPU-abc"]
    assert result["uuid"] == "GPU-abc"
    assert result["utilization_pct"] == 72.
    assert result["memory_used_mib"] == 100.
