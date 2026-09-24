"""LEGACY RESEARCH SCRIPT: retained for extraction or removal review; not part of the public Sleepwalker API."""

import os 
import torch
import tqdm
import hashlib

from dotenv import load_dotenv
from os.path import join
from functools import partial
from pyedflib import EdfReader

from sleepwalker.models.MaskedAutoencoder import MaskedAutoencoder
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.datasets.Ruhrlandklinik import Ruhrlandklinik
from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.trainer.MaskedAutoencoderTrainer import MaskedAutoencoderTrainer
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.utils.targets import build_multitask_target, normalize_multitask_config
from sleepwalker.utils import logger, MlflowSink, count_parameters

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

load_dotenv()

SAMPLE_FREQUENCY=float(os.environ.get('SAMPLE_FREQUENCY', 100.0))
BATCH_SIZE=int(os.environ.get('BATCH_SIZE', 128))
N_TRAIN_SAMPLES=int(os.environ.get('N_TRAIN_SAMPLES', 50_000))
N_VAL_SAMPLES=int(os.environ.get('N_VAL_SAMPLES', 15_000))
N_TEST_SAMPLES=int(os.environ.get('N_TEST_SAMPLES', 25_000))
N_WORKERS_DATASET=int(os.environ.get('N_WORKERS_DATASET', 24))
N_WORKERS_DATALOADER=int(os.environ.get('N_WORKERS_DATALOADER', 24))
SUBSAMPLE_WINDOW_PERCENT = None
DEVICE=os.environ.get('DEVICE', 'cuda')

CHANNELS = [
    ChannelConfig('EEG', ['C4-M1', 'F4-M1', 'O2-M1', 'C3-M2', 'F3-M2', 'O1-M2'], EEGFilterNormalizer(fs=SAMPLE_FREQUENCY)),
    ChannelConfig('EOG', ['E1-M2', 'E2-M1'], EEGFilterNormalizer(fs=SAMPLE_FREQUENCY)),
    ChannelConfig('RIP', ['RIP Flow', 'RIP Sum', 'Abdomen', 'Chest'], SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=10.0, band_order=4, notch_freq=None)),
    ChannelConfig('SPO2', ['Saturation'], SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=10.0, band_order=4, notch_freq=None)),
    ChannelConfig('LEG-EMG', ['Left Leg', 'Right Leg'], SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=10.0, highcut=45.0, band_order=4, notch_freq=None)),
    ChannelConfig('ECG', ['ECG'], SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=45)),
]

EVENT_MAPPING = {
    'wach': 'wake',
    'n1': 'n1',
    'n2': 'n2',
    'n3': 'n3',
    'rem': 'rem',
    'arousal': 'arousal',
    'a. gemischt': 'apnea',
    'a. obstruktiv': 'apnea',
    'a. zentral': 'apnea',
    'apnoe': 'apnea',
    'h. obstruktiv': 'hypopnea',
    'h. zentral': 'hypopnea',
    'hypopnea-gemischt': 'hypopnea',
    'hypopnoe': 'hypopnea',
    'lm': 'lm',
}

def prepare_sleep_staging_patient(data_df, label_df, label_extra_df, **_kwargs):
    # Remove PAP patients
    # original_labels = EdfReader(_kwargs['patient']).getSignalLabels()
    # if 'Druck (PAP)' in original_labels or 'EPAP' in original_labels or 'IPAP' in original_labels or 'Fluss (PAP)' in original_labels:
    #     return None
    trimmed = trim_event(data_df, label_df, label_extra_df, keep_events=['wake'])
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    if label_df is not None and SUBSAMPLE_WINDOW_PERCENT:
        label_df = label_df.groupby('Label', group_keys=False).sample(frac=SUBSAMPLE_WINDOW_PERCENT)
        if label_extra_df:
            label_extra_df = label_extra_df.groupby('Label', group_keys=False).sample(frac=SUBSAMPLE_WINDOW_PERCENT)

    return data_df, label_df, label_extra_df

def get_collate_ignore_list(dataset) -> list[str]:
    return ['time', 'dataset', 'target', 'data'] if isinstance(dataset, MultiDataset) else ['time', 'target', 'data']

def prepare_multiclass_sample(data, target, task_config, **item):
    new_item = {}
    new_item['patient'] = torch.tensor(int(hashlib.sha256(item['patient'].encode('utf-8')).hexdigest(), 16) % 10**8).long()

    # Build individual data modalities
    for modality_name in dict.fromkeys(data.columns):  
        n_expected = len([ch.name for ch in CHANNELS if ch.group == modality_name])
        modality_df = data.loc[:, data.columns == modality_name]
        n_present = len(modality_df.columns)

        mask = torch.ones((n_expected))
        raw_values = torch.from_numpy(modality_df.values).float()

        pad = n_expected - n_present
        if pad > 0:
            raw_values = torch.cat([raw_values, torch.zeros((raw_values.shape[0], pad))], dim=-1)
            mask[-pad:] = 0

        new_item[f'data_{modality_name}'] = raw_values
        new_item[f'mask_{modality_name}'] = mask

    # Build all different target annotations
    targets, _ = build_multitask_target(target, task_config)
    for task_idx, (task_name, cfg) in enumerate(task_config.items()):
        task_targets = targets[task_idx, :cfg['n_steps'], :len(cfg['labels'])]
        new_item[f'target_{task_name}'] = task_targets[len(task_targets) // 2]

    return new_item

task_config = {
    'breathing': {
        'labels': ['apnea', 'hypopnea', 'regular'],
        'default': 'regular',
        'percentage': 0.5,
        'target_resolution': '10s',
        'sequence_len': 3,
        'embeddings': ['RIP', 'SPO2'],
        'n_slices': 50,
        'class_weights': {0: 4, 1: 4, 2: 1},
        'loss_function': torch.nn.functional.cross_entropy,
        'loss_mode': 'inverse',
    },
    'legmovement': {
        'labels': ['lm', 'no lm'],
        'default': 'no lm',
        'percentage': 0.5,
        'target_resolution': '1s',
        'sequence_len': 30,
        'embeddings': ['LEG-EMG'],
        'class_weights': {0: 10, 1: 1},
        'n_slices': 10,
        'loss_function': torch.nn.functional.cross_entropy,
        'loss_mode': 'inverse',
    },
    # 'arousal': {
    #     'labels': ['arousal', 'no arousal'],
    #     'default': 'no arousal',
    #     'percentage': 0.5,
    #     'target_resolution': '1s',
    #     'embeddings': ['EEG'],
    #     'class_weights': {0: 10, 1: 1},
    #     'n_slices': 10,
    #     'loss_function': torch.nn.functional.cross_entropy,
    #     'loss_mode': 'inverse',
    # },
    # 'desat': {
    #     'labels': ['desaturation', 'no desaturation'],
    #     'default': 'no desaturation',
    #     'percentage': 0.5,
    #     'target_resolution': '10s',
    #     'loss_function': torch.nn.functional.cross_entropy,
    #     'loss_mode': 'inverse',
    # },
    'sleep': {
        'labels': ['n1', 'n2', 'n3', 'rem', 'wake'],
        'default': None,
        'percentage': 0.5,
        'target_resolution': '30s',
        'sequence_len': 1,
        'embeddings': ['EEG', 'EOG'],
        'n_slices': 50,
        'loss_function': torch.nn.functional.cross_entropy,
        'loss_mode': 'inverse',
    },
}
normalized_task_config = normalize_multitask_config(task_config)

def get_datasets():
    dataset_path = join(os.environ['DATASET_DIR'], 'ruhrlandklinik/raw/train-test-2023')
    all_patients = get_edf_files_in_repo(dataset_path, recursive=False)
    train_patients, rest = random_split(all_patients, test_frac=0.33, seed=1912817)
    val_patients, test_patients = random_split(rest, test_frac=0.5, seed=918171)

    logger.context('TRAIN')
    train_ds = Ruhrlandklinik(
        channels=CHANNELS, 
        event_mapping=EVENT_MAPPING,
        sample_frequency=SAMPLE_FREQUENCY,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_sample=partial(prepare_multiclass_sample, task_config=normalized_task_config),
        total_input='330s',
        target_resolution='30s',
        return_nox=False,
        group_sampling_strategy='none',
    )
    train_ds.initialize(patients=train_patients, num_workers=N_WORKERS_DATASET)
    logger.uncontext()

    logger.context('VAL')
    val_ds = Ruhrlandklinik(
        channels=CHANNELS, 
        event_mapping=EVENT_MAPPING,
        sample_frequency=SAMPLE_FREQUENCY,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_sample=partial(prepare_multiclass_sample, task_config=normalized_task_config),
        total_input='330s',
        target_resolution='30s',
        return_nox=False,
        group_sampling_strategy='none',
    )
    val_ds.initialize(patients=val_patients, num_workers=N_WORKERS_DATASET)
    logger.uncontext()

    logger.context('TEST')
    test_ds = Ruhrlandklinik(
        channels=CHANNELS, 
        event_mapping=EVENT_MAPPING,
        sample_frequency=SAMPLE_FREQUENCY,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_sample=partial(prepare_multiclass_sample, task_config=normalized_task_config),
        total_input='330s',
        target_resolution='30s',
        return_nox=False,
        group_sampling_strategy='none',
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
        pin_memory=True
    )
    test_sampler = torch.utils.data.RandomSampler(test_ds, num_samples=min(len(test_ds), N_TEST_SAMPLES)) 
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        sampler=test_sampler,
        num_workers=N_WORKERS_DATALOADER, 
        collate_fn=partial(batch_collate, ignore_list=get_collate_ignore_list(test_ds)), 
        drop_last=False, 
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def main():

    train_ds, val_ds, test_ds = get_datasets()
    train_dl, val_dl, test_dl = get_dataloader(train_ds, val_ds, test_ds)

    model = MaskedAutoencoder(
        groups=train_ds.channel_groups,
        normalize=False,
    )
    print('Number of parameters:', count_parameters(model))
    model_hp = model.get_hyperparameters()

    print('Tracking to MLFLOW instance', os.environ['MLFLOW_URL'])
    logger.add_sink(MlflowSink(tracking_uri=os.environ['MLFLOW_URL'], experiment='debug', artifact_uri=None))
    logger.start_run(run_name='testrun-full-nonormalize-10pPatients', params=model_hp)

    EPOCHS=31
    trainer = MaskedAutoencoderTrainer(
        groups=train_ds.channel_groups,
        epochs=EPOCHS,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=2e-3),
        #lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=EPOCHS),
        classes=train_ds.get_classes(),
        downstream_tasks=normalized_task_config,
        save_every=10,
        device=DEVICE,
        warmup_device='cpu',
    )

    trainer.fit(model, train_loader=train_dl, val_loader=val_dl, test_loader=test_dl)

if __name__ == '__main__':
    main()
