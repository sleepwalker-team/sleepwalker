from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
import yaml

from sleepwalker.cli import train
from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import load_embedding_model, load_packaged_model, save_packaged_model
from sleepwalker.models.BatchClassifierModel import BatchClassifierModel
from sleepwalker.models.PreprocessorEmbeddingModel import PreprocessorEmbeddingModel
from sleepwalker.models.batch.LightGBMModel import LightGBMModel
import sleepwalker.models.DRIVEN as driven_module
from sleepwalker.trainer.BaseTrainer import BaseTrainer
from sleepwalker.trainer.BatchTrainer import BatchTrainer
from sleepwalker.trainer.utils.disk import load_checkpoint
from sleepwalker.training.loader import build_loader
from sleepwalker.models.preprocessors.EmpiricalClipScaler import EmpiricalClipScaler


CLASSES = ["regular", "event"]


class ProbabilityEstimator:
    def __init__(self, classes=CLASSES):
        self.classes = list(classes)
        self.fitted = False
        self.fit_calls = 0
        self.validation = None
        self.seen = None

    def fit(self, X, y, validation_data=None):
        self.fit_calls += 1
        self.seen = (np.array(X), np.array(y))
        self.validation = None if validation_data is None else tuple(np.array(part) for part in validation_data)
        self.fitted = True

    def is_fitted(self):
        return self.fitted

    def probability_classes(self):
        return list(reversed(self.classes))

    def predict_proba(self, X):
        positive = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack([positive, 1 - positive])


class TrainingMean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("count", torch.zeros(()))

    def requires_warmup(self):
        return True

    def update(self, data):
        self.mean.add_(data.sum())
        self.count.add_(data.numel())

    def forward(self, data):
        return data - self.mean / self.count


class TinyTemporalBranch(torch.nn.Module):
    """Cheap branch for native lifecycle coverage; real topology has its own tests."""

    def __init__(self, dropout, image_scale, image_offset):
        super().__init__()
        self.projection = torch.nn.Linear(1, 1280)
        self.head = torch.nn.Linear(1280, 2)

    def forward(self, signal):
        pooled = self.projection(signal.mean(dim=1, keepdim=True)).tanh()
        return self.head(pooled), pooled


def features(steps=None):
    return PreprocessorEmbeddingModel(ts_len=2, n_channels=1, embedding_dim=2, preprocessors=[*(steps or []), torch.nn.Flatten(1)])


def batch(values=(-2, -1, 1, 2)):
    data = torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1).repeat(1, 2, 1)
    labels = (data[:, 0, 0] > 0).long()
    return {"data": data, "target": torch.nn.functional.one_hot(labels, 2).float().unsqueeze(1), "patient": [f"patient-{index}" for index in range(len(values))], "time": list(range(len(values)))}


def trainer(tmp_path=None, **kwargs):
    if tmp_path is not None:
        kwargs.update(store_on_disk=True, feature_path=tmp_path / "features")
    return BatchTrainer(CLASSES, "1s", **kwargs)


def classifier(estimator=None, encoder=None):
    return BatchClassifierModel(features() if encoder is None else encoder, ProbabilityEstimator() if estimator is None else estimator, CLASSES)


def inference_dataset():
    return UnlabelledDataset(channels=[ChannelConfig("signal", ["signal"])], sample_frequency=2, total_input="1s", stride="1s")


@pytest.mark.parametrize("disk", [False, True])
def test_batch_fit_preserves_rows_masks_validation_and_probability_order(tmp_path, disk):
    training = batch()
    training["target_mask"] = torch.tensor([[True], [False], [True], [True]])
    current = trainer(tmp_path if disk else None)
    estimator = ProbabilityEstimator()
    result = classifier(estimator)
    current.fit(result, [training], [batch((10, 20))])
    assert not isinstance(current, BaseTrainer)
    assert isinstance(result, BatchClassifierModel)
    np.testing.assert_array_equal(estimator.seen[0][:, 0], [-2, 1, 2])
    np.testing.assert_array_equal(estimator.seen[1], [0, 1, 1])
    np.testing.assert_array_equal(estimator.validation[0][:, 0], [10, 20])
    probabilities = result(training["data"]).softmax(-1)
    assert probabilities.shape == (4, 1, 2)
    assert probabilities[0, 0, 0] > probabilities[0, 0, 1]
    assert probabilities[-1, 0, 1] > probabilities[-1, 0, 0]
    assert current.fit(result, [])["feature_extraction"] == current.extraction_info
    assert estimator.fit_calls == 1


def test_validation_none_and_supplied_empty_validation_are_distinct():
    estimator = ProbabilityEstimator()
    trainer().fit(classifier(estimator), [batch()])
    assert estimator.validation is None
    with pytest.raises(ValueError, match="no usable"):
        trainer().fit(classifier(), [batch()], [])


def test_capped_loader_selection_has_bounded_bookkeeping_and_deterministic_order():
    options = dict(batch_size=4, num_workers=0, n_samples=20, collate_fn=list, sampling="random")
    indices = list(build_loader(range(10**12), seed=17, **options).sampler)
    assert len(indices) == len(set(indices)) == 20
    assert all(0 <= value < 10**12 for value in indices)
    assert indices == list(build_loader(range(10**12), seed=17, **options).sampler)
    assert indices != list(build_loader(range(10**12), seed=18, **options).sampler)


def test_feature_preparation_fits_only_training_and_reuse_preserves_statistics():
    pipeline = features([TrainingMean()])
    current = trainer(prepare_features="fit")
    current.fit(classifier(encoder=pipeline), [batch((1, 3))], [batch((100, 200))])
    step = pipeline.preprocessors[0]
    assert step.count == 4 and step.mean == 8
    trainer(prepare_features="reuse").fit(classifier(encoder=pipeline), [batch((100, 200))])
    assert step.count == 4 and step.mean == 8


def test_saved_preprocessor_statistics_follow_module_dtype():
    step = EmpiricalClipScaler()
    step.update(torch.tensor([[[-1.], [1.]], [[-2.], [2.]]]))
    step.to(dtype=torch.float64)
    assert step.mins.dtype == step.maxs.dtype == torch.float64


@pytest.mark.parametrize("invalid", ["soft", "temporal", "weights", "shape"])
def test_invalid_batch_target_contract_fails(invalid):
    current = batch()
    if invalid == "soft":
        current["target"][:] = 0.5
    elif invalid == "temporal":
        current["target"] = current["target"].repeat(1, 2, 1)
    elif invalid == "weights":
        current["sample_weight"] = torch.ones(4)
    else:
        current["target"] = current["target"][:-1]
    with pytest.raises(ValueError):
        trainer().fit(classifier(), [current])


def test_interrupted_disk_extraction_restarts_and_overwrites_partial_arrays(tmp_path):
    current = trainer(tmp_path, checkpoint_path=tmp_path / "checkpoint.pt")

    def interrupted():
        yield batch((-2, 1))
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        current.fit(classifier(), interrupted())
    restored, result = load_checkpoint(tmp_path / "checkpoint.pt")
    restored.fit(result, [batch((-3, 3)), batch((-1, 2))])
    np.testing.assert_array_equal(result.batch_model.seen[0][:, 0], [-3, 3, -1, 2])
    assert restored.extraction_info["train"]["rows"] == 4
    restored, result = load_checkpoint(tmp_path / "checkpoint.pt")
    assert restored.fit(result, [])["feature_extraction"] == restored.extraction_info


def test_relocated_composite_owns_loaded_encoder_without_source_or_store(tmp_path):
    dataset = inference_dataset()
    source = tmp_path / "source"
    save_packaged_model(source, name="encoder", model=features(), dataset=dataset)
    encoder = load_embedding_model(source)
    current = trainer(tmp_path)
    combined = classifier(encoder=encoder)
    current.fit(combined, [batch()])
    destination = tmp_path / "final"
    save_packaged_model(destination, name="hybrid", task="breathing", model=combined, dataset=dataset, classification_contract=current.classification_contract())
    expected = combined(batch()["data"])
    shutil.rmtree(source)
    shutil.rmtree(tmp_path / "features")
    moved = tmp_path / "relocated"
    destination.rename(moved)
    loaded = load_packaged_model(moved)
    torch.testing.assert_close(loaded.model(batch()["data"]), expected)
    assert loaded.capabilities == ["embeddings", "classification"]
    assert loaded.model.features(batch()["data"]).shape == (4, 2)
    loaded.model.train()
    assert not loaded.model.embedding_model.training


def test_lightgbm_memory_and_disk_training_agree(tmp_path):
    pytest.importorskip("lightgbm")

    results = []
    for disk in [False, True]:
        model = LightGBMModel(CLASSES, parameters={"num_threads": 1, "min_data_in_leaf": 1, "num_leaves": 3, "seed": 17, "force_col_wise": True}, num_boost_round=4, early_stopping_rounds=2)
        trained = classifier(model)
        trainer(tmp_path if disk else None).fit(trained, [batch()], [batch((-3, 3))])
        assert model.validation_used
        results.append(trained(batch()["data"]))
    torch.testing.assert_close(*results)


@pytest.mark.parametrize("neural_kind", ["single", "driven"])
def test_native_gradient_then_batch_export_and_resume_without_source_package(tmp_path, monkeypatch, neural_kind):
    pytest.importorskip("lightgbm")
    classes = ["wake", "n1", "n2", "n3", "rem"]
    fixture = Path(__file__).parent / "data"
    data = {
        "name": "sleepwalker.datasets.SyntheticDataset.SyntheticDataset",
        "channels": [{"logical_name": "signal", "physical_names": ["EEG"], "preprocessors": [{"name": "sleepwalker.datasets.normalizer.ConvertUnit.ConvertUnit", "target": "uV"}]}],
        "sample_frequency": 2, "total_input": "1s", "stride": "1s", "annotation_labels": classes, "output_classes": classes,
        "prepare_target": {"name": "sleepwalker.trainer.utils.targets.prepare_multiclass_target", "target_classes": classes, "target_resolution": "1s"},
        "files": {"train": [str(fixture / "signals_01.edf")], "validation": [str(fixture / "signals_02.edf")], "test": [str(fixture / "signals_03.edf")]},
        "num_workers": 0, "strict": True,
    }
    neural = tmp_path / "neural"
    config = {"seed": 17, "data": data,
        "model": {"name": "sleepwalker.models.UTime.UTime", "channel": [2, 4], "kernel": [3, 3], "maxpool": [2, 1], "mlp_size": 4, "sampling_frequency": 2},
        "trainer": {"name": "sleepwalker.trainer.MulticlassTrainer.MulticlassTrainer", "epochs": 1, "optimizer": {"name": "torch.optim.Adam", "lr": 0.001}, "loss_function": "torch.nn.functional.cross_entropy", "target_resolution": "1s", "eval_every": 1, "save_every": 1, "device": "cpu"},
        "run": {"experiment_name": "neural-smoke", "batch_size": 2, "n_samples": 4, "n_samples_test": 4, "num_workers_dataloader": 0, "log_path": str(tmp_path / "runs"), "package_path": str(neural), "expert_task": "sleep"}}
    if neural_kind == "driven":
        monkeypatch.setattr(driven_module, "EfficientNetV2Branch", TinyTemporalBranch)
        classes = ["regular", "respiratory_event"]
        tasks = {name: {"labels": classes, "default": "regular", "sequence_len": 1, "target_resolution": "30s", "percentage": 1 / 60, "loss_function": "torch.nn.functional.cross_entropy"} for name in ["abd", "spo2_0", "spo2_10", "spo2_15", "spo2_20", "spo2_25"]}
        data.update(sample_frequency=2, total_input="55s", stride="15s", annotation_labels=["sleep", "respiratory_event"], output_classes=classes)
        data["channels"].append({"logical_name": "spo2", "physical_names": ["EEG"]})
        data["prepare_events"] = {"name": "sleepwalker.training.callbacks.map_events", "mapping": {"wake": "sleep", "n1": "sleep", "n2": "sleep", "n3": "sleep", "rem": "respiratory_event"}}
        data["prepare_target"] = {"name": "sleepwalker.trainer.utils.targets.prepare_multitask_target", "task_config": tasks}
        config["model"] = {"name": "sleepwalker.models.DRIVEN.DRIVEN", "task_config": tasks}
        config["trainer"] = {"name": "sleepwalker.trainer.MultiLabelTrainer.MultiLabelTrainer", "task_config": tasks, "epochs": 1, "optimizer": {"name": "torch.optim.Adam", "lr": 0.001}, "eval_every": 1, "save_every": 1, "device": "cpu", "balance_batches": True}
        config["run"]["n_samples"] = 64
    path = tmp_path / "train-neural.yml"
    path.write_text(yaml.safe_dump(config))
    train.main(["train", str(path)])
    assert (neural / "manifest.json").is_file()
    if neural_kind == "driven":
        data["prepare_target"] = {"name": "sleepwalker.trainer.utils.targets.prepare_multiclass_target", "target_classes": classes, "target_resolution": "30s", "target_offset": "0s", "percentage": 1 / 60}
    config["model"] = {"name": "sleepwalker.models.BatchClassifierModel.BatchClassifierModel", "embedding_model": {"name": "sleepwalker.deployment.load_embedding_model", "package": str(neural)}, "batch_model": {"name": "sleepwalker.models.batch.LightGBMModel.LightGBMModel", "num_boost_round": 2, "parameters": {"num_threads": 1, "min_data_in_leaf": 1, "num_leaves": 3}}}
    config["trainer"] = {"name": "sleepwalker.trainer.BatchTrainer.BatchTrainer", "target_resolution": "30s" if neural_kind == "driven" else "1s", "store_on_disk": True, "feature_path": str(tmp_path / "features")}
    config["run"].update(experiment_name="batch-smoke", n_samples=16, n_samples_validation=16, train_sampling="random", validation_sampling="random", package_path=str(tmp_path / "hybrid"))
    path = tmp_path / "train-batch.yml"
    path.write_text(yaml.safe_dump(config))
    train.main(["train", str(path)])
    shutil.rmtree(neural)
    shutil.rmtree(tmp_path / "features")
    package = load_packaged_model(tmp_path / "hybrid")
    assert package.model(torch.ones(2, 110, 2) if neural_kind == "driven" else torch.ones(2, 2, 1)).shape == (2, 1, len(classes))
    restored = train.runcfg_from_checkpoint(tmp_path / "runs/batch-smoke/batch/checkpoint.pt")
    assert restored.model.batch_model.is_fitted()
    assert not hasattr(restored, "embedding_model")


def test_unfitted_composite_can_be_constructed_but_not_used_or_exported(tmp_path):
    model = classifier()
    assert model.features(batch()["data"]).shape == (4, 2)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model(batch()["data"])
    with pytest.raises(RuntimeError, match="not been fitted"):
        save_packaged_model(tmp_path / "unfitted", name="unfitted", task="breathing", model=model, dataset=inference_dataset(), classification_contract=trainer().classification_contract())
