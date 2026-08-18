import os 
import torch
import hashlib

from dotenv import load_dotenv
from os.path import join
from functools import partial

from sleepwalker.models.MaskedAutoencoder import MaskedAutoencoder
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.datasets.HSP import HSP
from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.trainer.MaskedAutoencoderTrainer import MaskedAutoencoderTrainer
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.utils import logger, WandbSink, count_parameters

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

load_dotenv()

SAMPLE_FREQUENCY=float(os.environ.get('SAMPLE_FREQUENCY', 100.0))
BATCH_SIZE=int(os.environ.get('BATCH_SIZE', 96))
N_TRAIN_SAMPLES=int(os.environ.get('N_TRAIN_SAMPLES', 250_000))
N_WARMUP_SAMPLES=int(os.environ.get('N_WARMUP_SAMPLES', 50_000))
N_VAL_SAMPLES=int(os.environ.get('N_VAL_SAMPLES', 15_000))
N_TEST_SAMPLES=int(os.environ.get('N_TEST_SAMPLES', 25_000))
N_WORKERS_DATASET=int(os.environ.get('N_WORKERS_DATASET', 24))
N_WORKERS_DATALOADER=int(os.environ.get('N_WORKERS_DATALOADER', 24))
N_PATIENTS = 1000
SUBSAMPLE_WINDOW_PERCENT = None
DEVICE=os.environ.get('DEVICE', 'cuda')

CHANNELS = [
    ChannelConfig(name='C4-M1', normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY), group='EEG'),
    ChannelConfig(name='F4-M1', normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY), group='EEG'),
    ChannelConfig(name='O2-M1', normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY), group='EEG'),
    ChannelConfig(name='C3-M2', normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY), group='EEG'),
    ChannelConfig(name='F3-M2', normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY), group='EEG'),
    ChannelConfig(name='O1-M2', normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY), group='EEG'),
    ChannelConfig(name='E1-M2', normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY), group='EOG'),
    ChannelConfig(name='E2-M1', normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY), group='EOG'),
    ChannelConfig(name='ABD', normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=10.0, band_order=4, notch_freq=None), group='RESP'),
    ChannelConfig(name='CHEST', normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=10.0, band_order=4, notch_freq=None), group='RESP'),
    ChannelConfig(name='SaO2', normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=10.0, band_order=4, notch_freq=None), group='RESP'), 
    ChannelConfig(name='SpO2', normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=10.0, band_order=4, notch_freq=None), group='RESP'), 
    ChannelConfig(name='IC', normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=10.0, highcut=40.0, band_order=4, notch_freq=None), group='RESP'), 
    ChannelConfig(name='PTAF', normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=15.0, band_order=4, notch_freq=None), group='RESP'), 
    ChannelConfig(name='AIRFLOW', normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=15.0, band_order=4, notch_freq=None), group='RESP'), 
    # ChannelConfig(name='LAT', normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=10.0, highcut=45.0, band_order=4, notch_freq=None), group='LEG-EMG'), 
    # ChannelConfig(name='RAT', normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=10.0, highcut=45.0, band_order=4, notch_freq=None), group='LEG-EMG'), 
    # ChannelConfig(name='EKG', normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=45), group='ECG'), 
]

EVENT_MAPPING = {
    'wake': 'wake',
    'n1': 'n1',
    'n2': 'n2',
    'n3': 'n3',
    'rem': 'rem',
    'obstructive-apnea': 'apnea',
    'central-apnea': 'apnea',
    'mixed-apnea': 'apnea',
    'hypopnea': 'hypopnea',
    'right leg': 'lm',
    'left leg': 'lm',
    'leg movement': 'lm',
    'leg movements': 'lm',
}

task_config = {
    'breathing': {
        'labels': ['apnea', 'hypopnea', 'regular'],
        'default': 'regular',
        'percentage': 0.5,
        'target_resolution': '10s',
        'embeddings': ['RESP'],
        'n_slices': 50,
        'class_weights': {0: 4, 1: 4, 2: 1},
        'loss_function': torch.nn.functional.cross_entropy,
        'loss_mode': 'inverse',
    },
    # 'legmovement': {
    #     'labels': ['lm', 'no lm'],
    #     'default': 'no lm',
    #     'percentage': 0.5,
    #     'target_resolution': '1s',
    #     'embeddings': ['LEG-EMG'],
    #     'class_weights': {0: 10, 1: 1},
    #     'n_slices': 10,
    #     'loss_function': torch.nn.functional.cross_entropy,
    #     'loss_mode': 'inverse',
    # },
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
        'default': 'wake',
        'percentage': 0.5,
        'target_resolution': '30s',
        'embeddings': ['EEG', 'EOG'],
        'n_slices': 50,
        'loss_function': torch.nn.functional.cross_entropy,
        'loss_mode': 'inverse',
    },
}
normalized_task_config = MultiLabelTrainer.normalize_task_config(task_config)

def assure_all_groups_present(data_df):
    groups = {group_name: [cc.name for cc in CHANNELS if cc.group == group_name] for group_name in set([c.group for c in CHANNELS])}
    for group_name, group_channels in groups.items():
        if not any([c in data_df.columns for c in group_channels]):
            print('Patient has no channel for group', group_name)
            return None
    return data_df

def subsample_patient_windows(label_df, label_extra_df):
    if SUBSAMPLE_WINDOW_PERCENT is not None:
        label_df = label_df.groupby('Label', group_keys=False).sample(frac=SUBSAMPLE_WINDOW_PERCENT)
        if label_extra_df:
            label_extra_df = label_extra_df.groupby('Label', group_keys=False).sample(frac=SUBSAMPLE_WINDOW_PERCENT)

    return label_df, label_extra_df

def prepare_sleep_staging_patient(data_df, label_df, label_extra_df, patient, **_kwargs):

    edf_duration_seconds = len(data_df) / SAMPLE_FREQUENCY
    label_duration_seconds = (label_df['Endtime'].iloc[-1] - label_df['Starttime'].iloc[0]).seconds
    duration_hours = min(edf_duration_seconds, label_duration_seconds) / 3600
    if duration_hours <= 4.0:
        print(f'Skip patient {patient} for a too short reading: {duration_hours:.2f}h')
        return None

    trimmed = trim_event(data_df, label_df, label_extra_df, keep_events=['n1', 'n2', 'n3', 'rem'])
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed

    data_df = assure_all_groups_present(data_df)
    if data_df is None:
        return None

    label_df, label_extra_df = subsample_patient_windows(label_df, label_extra_df)

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
    t = MultiLabelTrainer.build_multitask_target(target, task_config)
    for task_name, cfg, labels in zip(task_config.keys(), task_config.values(), t):
        labels = labels[:cfg['n_steps']]
        # Take middle slice
        middle = len(labels) // 2
        _target = torch.zeros((len(cfg['labels'])))
        _target[int(labels[middle])] = 1
        new_item[f'target_{task_name}'] = _target.float()

    return new_item

def get_datasets():
    dataset_path = join(os.environ['DATASET_DIR'], 'hsp')
    all_patients = get_edf_files_in_repo(dataset_path, recursive=True)[:N_PATIENTS]
    train_patients, rest = random_split(all_patients, test_frac=0.3, seed=1912817)
    val_patients, test_patients = random_split(rest, test_frac=0.5, seed=918171)

    logger.context('TRAIN')
    train_ds = HSP(
        channels=CHANNELS,
        event_mapping=EVENT_MAPPING,
        sample_frequency=SAMPLE_FREQUENCY,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_sample=partial(prepare_multiclass_sample, task_config=normalized_task_config),
        total_input='330s',
        target_resolution='30s',
        group_sampling_strategy='none',
    )
    train_ds.initialize(patients=train_patients, num_workers=N_WORKERS_DATASET)
    logger.uncontext()

    logger.context('VAL')
    val_ds = HSP(
        channels=CHANNELS,
        event_mapping=EVENT_MAPPING,
        sample_frequency=SAMPLE_FREQUENCY,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_sample=partial(prepare_multiclass_sample, task_config=normalized_task_config),
        total_input='330s',
        target_resolution='30s',
        group_sampling_strategy='none',
    )
    val_ds.initialize(patients=val_patients, num_workers=N_WORKERS_DATASET)
    logger.uncontext()

    logger.context('TEST')
    test_ds = HSP(
        channels=CHANNELS,
        event_mapping=EVENT_MAPPING,
        sample_frequency=SAMPLE_FREQUENCY,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_sample=partial(prepare_multiclass_sample, task_config=normalized_task_config),
        total_input='330s',
        target_resolution='30s',
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
        persistent_workers=True, 
        prefetch_factor=8, 
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
    warmup_sampler = torch.utils.data.RandomSampler(train_ds, num_samples=min(len(train_ds), N_WARMUP_SAMPLES)) 
    warmup_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=warmup_sampler,
        num_workers=N_WORKERS_DATALOADER, 
        collate_fn=partial(batch_collate, ignore_list=get_collate_ignore_list(train_ds)), 
        drop_last=False, 
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, warmup_loader

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=0.5, min_lr_ratio=0.0, last_epoch=-1):
    
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step):
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def setup_wandb_metrics(sink):
    sink.wandb.define_metric('batch_train_step')
    sink.wandb.define_metric('batch_val_step')
    sink.wandb.define_metric('batch/loss/train', step_metric='batch_train_step')
    sink.wandb.define_metric('batch/loss/val', step_metric='batch_val_step')

    sink.wandb.define_metric('epoch_step')
    sink.wandb.define_metric('epoch/train/*', step_metric='epoch_step')
    sink.wandb.define_metric('epoch/val/*', step_metric='epoch_step')

    for task_name in task_config:
        sink.wandb.define_metric(f'epoch/{task_name}/f1_micro', step_metric='epoch_step')
        sink.wandb.define_metric(f'epoch/{task_name}/f1_macro', step_metric='epoch_step')
        sink.wandb.define_metric(f'epoch/{task_name}/accuracy', step_metric='epoch_step')
        sink.wandb.define_metric(f'epoch/{task_name}/coehns_kappa', step_metric='epoch_step')

def main():

    train_ds, val_ds, test_ds = get_datasets()
    train_dl, val_dl, test_dl, warmup_dl = get_dataloader(train_ds, val_ds, test_ds)

    model = MaskedAutoencoder(
        groups=train_ds.channel_groups,
    )
    print('Number of parameters:', count_parameters(model))
    model_hp = model.get_hyperparameters()

    print('Tracking to WandB instance', os.environ['WANDB_BASE_URL'])
    sink = WandbSink(tracking_uri=os.environ['WANDB_BASE_URL'], experiment='mae-hsp', artifact_uri=None)
    logger.add_sink(sink)
    logger.start_run(run_name='1k', params=model_hp)
    setup_wandb_metrics(sink)

    EPOCHS=20
    steps_per_epoch = N_TRAIN_SAMPLES // BATCH_SIZE 
    num_training_steps = steps_per_epoch * EPOCHS
    num_warmup_steps = int(0.05 * num_training_steps) 

    trainer = MaskedAutoencoderTrainer(
        groups=train_ds.channel_groups,
        epochs=EPOCHS,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=2e-3),
        #lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=EPOCHS),
        #lr_scheduler = lambda optimizer: get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps),
        classes=train_ds.get_classes(),
        downstream_tasks=normalized_task_config,
        save_every=30,
        device=DEVICE,
        warmup_device='cpu',
    )

    trainer.fit(model, train_loader=train_dl, val_loader=val_dl, test_loader=test_dl, warmup_loader=warmup_dl)

if __name__ == '__main__':
    main()
