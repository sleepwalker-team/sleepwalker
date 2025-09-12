import pandas as pd
import torch
from torchinfo import summary
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig
from sleepwalker.datasets.utils import get_edf_files_in_repo
from sleepwalker.datasets import Ruhrlandklinik

from sleepwalker.models import SleepTransformer
from sleepwalker.trainer.MulticlassTrainer import to_multiclass, MulticlassTrainer

def filter_patient(edf_path):
    meta = read_edf_meta(edf_path)
    return "C4-M1" in meta["signals"]

def get_item(item):
    try:
        freq = pd.to_timedelta(item["target"].index.freq).total_seconds()
        targets = torch.tensor(item["target"].sum().to_numpy())
        item["target"] = to_multiclass(targets, None, len(item["target"])*freq*0.5, True)

        if "target_extra" in item:
            freq = pd.to_timedelta(item["target_extra"].index.freq).total_seconds()
            targets = torch.tensor(item["target_extra"].sum().to_numpy())
            item["target_extra"] = to_multiclass(targets, None, len(item["target_extra"])*freq*0.5, True)
    
        return item
    except Exception as e:
        pass
    return None



#edf_folder = "/raid/projects/eeg-foundation/ruhrlandklinik/raw/"  # TODO: set your path
edf_folder = "/raid/data/ruhrlandklinik/raw/"
mode = "xval"                      # one of: xval, fixed, split, publish
n_splits = 5
patients_include = None            # e.g., ["patientA", "patientB"]
patients_exclude = None
batch_size = 128
total_input = "630s"
target_resolution = "30s"

all_patients = get_edf_files_in_repo(edf_folder, recursive=True)
all_patients = all_patients[:5]

dataset = Ruhrlandklinik(
    channels = [ChannelConfig(name="C4-M1", normalizer=None)],
    patients = all_patients,
    num_workers = 1,
    sample_frequency = 125,
    event_mapping = {
        "wach": "wake",
        "n1": "n1",
        "n2": "n2",
        "n3": "n3",
        "rem": "rem"
    },
    filter_patient = filter_patient,
    get_item = get_item,
    online_filtering = True,
    total_input = total_input, 
    target_resolution = target_resolution
)

#train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, sampler=None, shuffle=True, collate_fn=batch_collate, drop_last=True)

model = SleepTransformer(classes=dataset.get_classes(), n_channels=1)
model_stats = summary(model, input_size=(1, dataset.get_timeseries_len(), 1), depth=5, row_settings=["hide_recursive_layers"])

trainer = MulticlassTrainer(
    classes=dataset.get_classes(), 
    loss_function=torch.nn.functional.cross_entropy, 
    epochs=1, 
    batch_size=batch_size,
    optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=0.1)
)
trainer.fit(model, dataset)

