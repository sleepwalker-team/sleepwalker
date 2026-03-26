import os 
import torch
import tqdm

from dotenv import load_dotenv
from os.path import join
from functools import partial

from sleepwalker.models.MaskedAutoencoder import MaskedAutoencoder
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.datasets.Ruhrlandklinik import Ruhrlandklinik
from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.trainer.utils import trim_wake, get_target_as_multiclass
from sleepwalker.trainer.MaskedAutoencoderTrainer import MaskedAutoencoderTrainer
from sleepwalker.utils import logger, MlflowSink

load_dotenv()

SAMPLE_FREQUENCY=float(os.environ.get('SAMPLE_FREQUENCY', 100.0))
BATCH_SIZE=int(os.environ.get('BATCH_SIZE', 128))
N_TRAIN_SAMPLES=int(os.environ.get('N_TRAIN_SAMPLES', 50_000))
N_VAL_SAMPLES=int(os.environ.get('N_VAL_SAMPLES', 15_000))
N_WORKERS_DATASET=int(os.environ.get('N_WORKERS_DATASET', 24))
N_WORKERS_DATALOADER=int(os.environ.get('N_WORKERS_DATALOADER', 24))
DEVICE=os.environ.get('DEVICE', 'cuda')

def prepare_sleep_staging_patient(data_df, label_df, label_extra_df, **_kwargs):
    trimmed = trim_wake(data_df, label_df, label_extra_df)
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df

def get_collate_ignore_list(dataset) -> list[str]:
    return ['time', 'patient', 'dataset'] if isinstance(dataset, MultiDataset) else ['time', 'patient']

def prepare_multiclass_target(
    target,
    target_extra=None,
    percentage: float = 0.5,
    class_cnts=None,
    target_classes=None,
    **kwargs
):
    if target is None:
        return None

    if target_classes is not None:
        target = target.reindex(columns=target_classes, fill_value=0)
        if target_extra is not None:
            target_extra = target_extra.reindex(columns=target_classes, fill_value=0)

    return {
        'target': get_target_as_multiclass(
            target=target,
            target_extra=target_extra,
            percentage=percentage,
            class_cnts=class_cnts,
        )
    }

def prepare_multiclass_sample(data, target, **item):
    item['data'] = torch.from_numpy(data.values).float()
    item['target'] = torch.from_numpy(target.mean(axis=0).values).float()
    if 'target_extra' in item and item['target_extra'] is None:
        del item['target_extra']
    return item

def get_datasets():
    dataset_path = join(os.environ['DATASET_DIR'], 'ruhrlandklinik/raw/train-test-2023')
    all_patients = get_edf_files_in_repo(dataset_path, recursive=False)
    train_patients, rest = random_split(all_patients, test_frac=0.33, seed=1912817)
    val_patients, test_patients = random_split(rest, test_frac=0.5, seed=918171)

    channels = [
        ChannelConfig(name='C4-M1', normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY))
    ]
    event_mapping = {
        'wach': 'wake',
        'n1': 'n1',
        'n2': 'n2',
        'n3': 'n3',
        'rem': 'rem'
    }
    logger.context('TRAIN')
    train_ds = Ruhrlandklinik(
        channels=channels, 
        event_mapping=event_mapping,
        sample_frequency=SAMPLE_FREQUENCY,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_sample=prepare_multiclass_sample,
        total_input='330s',
        target_resolution='30s',
        return_nox=False,
    )
    train_ds.initialize(patients=train_patients, num_workers=N_WORKERS_DATASET)
    logger.uncontext()

    logger.context('VAL')
    val_ds = Ruhrlandklinik(
        channels=channels, 
        event_mapping=event_mapping,
        sample_frequency=SAMPLE_FREQUENCY,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_sample=prepare_multiclass_sample,
        total_input='330s',
        target_resolution='30s',
        return_nox=False,
    )
    val_ds.initialize(patients=val_patients, num_workers=N_WORKERS_DATASET)
    logger.uncontext()

    logger.context('TEST')
    test_ds = Ruhrlandklinik(
        channels=channels, 
        event_mapping=event_mapping,
        sample_frequency=SAMPLE_FREQUENCY,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_sample=prepare_multiclass_sample,
        total_input='330s',
        target_resolution='30s',
        return_nox=False,
    )
    test_ds.initialize(patients=test_patients, num_workers=N_WORKERS_DATASET)
    logger.uncontext()

    return train_ds, val_ds, test_ds

def get_dataloader(train_ds, val_ds, test_ds):
    train_sampler = torch.utils.data.RandomSampler(train_ds, num_samples=min(len(train_ds), N_TRAIN_SAMPLES)) 
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=N_WORKERS_DATALOADER, 
        collate_fn=partial(batch_collate, ignore_list=get_collate_ignore_list(train_ds)), 
        drop_last=False, 
        persistent_workers=True, 
        prefetch_factor=2, 
        pin_memory=True
    )

    val_sampler = torch.utils.data.RandomSampler(val_ds, num_samples=min(len(val_ds), N_VAL_SAMPLES)) 
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        sampler=val_sampler,
        num_workers=N_WORKERS_DATALOADER, 
        collate_fn=partial(batch_collate, ignore_list=get_collate_ignore_list(val_ds)), 
        drop_last=False, 
        persistent_workers=True, 
        prefetch_factor=2, 
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=N_WORKERS_DATALOADER, 
        collate_fn=partial(batch_collate, ignore_list=get_collate_ignore_list(test_ds)), 
        drop_last=False, 
        persistent_workers=True, 
        prefetch_factor=2, 
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def main():
    train_ds, val_ds, test_ds = get_datasets()
    train_dl, val_dl, test_dl = get_dataloader(train_ds, val_ds, test_ds)

    model = MaskedAutoencoder()

    print('Tracking to MLFLOW instance', os.environ['MLFLOW_URL'])
    logger.add_sink(MlflowSink(tracking_uri=os.environ['MLFLOW_URL'], experiment='debug', artifact_uri=None))
    logger.start_run(run_name='testrun', params={})

    trainer = MaskedAutoencoderTrainer(
        epochs=10,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=2e-3),
        classes=train_ds.get_classes(),
        device=DEVICE,
        warmup_device=DEVICE,
    )

    trainer.fit(model, train_loader=train_dl, val_loader=val_dl, test_loader=test_dl)

if __name__ == '__main__':
    main()