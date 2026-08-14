import json
from types import SimpleNamespace

import torch

import sleepwalker.deployment.package as package_module
from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import load_packaged_model
from sleepwalker.deployment.package import CloudpickleAdapter, sha256
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel
from tools.migrate_expert_package import migrate


class TinyClassifier(BaseModel, ClassifierModel):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 2)

    def compute(self, x):
        return self.head(x.mean(dim=1)).unsqueeze(1)

    def input_spec(self):
        return (1, 30, 1), {"layout": "BTC", "ts_len": 30, "n_channels": 1}


def test_v5_migration_is_an_explicit_one_time_step(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    dataset = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=1, total_input="30s", target_resolution="30s", stride="30s")
    old_expert = SimpleNamespace(name="old", task="sleep", model=TinyClassifier(), dataset=dataset, trainer=SimpleNamespace(classes=["wake", "sleep"], sequence_len=1), config={"fold": "fold_2"}, git_commit="abc")
    payload = source / "expert.pt"
    torch.save(old_expert, payload, pickle_module=CloudpickleAdapter)
    (source / "manifest.json").write_text(json.dumps({"format_version": "sleepwalker-expert-v5", "payload": payload.name, "sha256": sha256(payload)}), encoding="utf-8")

    destination = tmp_path / "new"
    migrate(source, destination)
    package = load_packaged_model(destination)

    assert package.name == "old"
    assert package.classification_contract == {"type": "single-head-multiclass", "classes": ["wake", "sleep"], "sequence_len": 1}
    assert package.config["fold"] == "fold_2"
    assert not hasattr(package_module, "Expert")
