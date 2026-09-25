"""LEGACY RESEARCH SCRIPT: retained for extraction or removal review; not part of the public Sleepwalker API."""

import os 
import torch
import tqdm
import hashlib
import torch.nn as nn
import numpy as np

from dotenv import load_dotenv
from os.path import join, exists
from functools import partial
from einops import rearrange

from sleepwalker.models.MaskedAutoencoder import MaskedAutoencoder
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split, export_dataloader_to_numpy_dir
from sleepwalker.datasets.Ruhrlandklinik import Ruhrlandklinik
from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.NumpyDataset import NumpyDataset
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.normalizer.RespirationFilterNormalizer import RespirationFilterNormalizer
from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer
from sleepwalker.trainer.utils import trim_event
from sleepwalker.trainer.MaskedAutoencoderTrainer import MaskedAutoencoderTrainer
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.utils.targets import build_multitask_target, normalize_multitask_config
from sleepwalker.utils import logger, MlflowSink, count_parameters
from os import makedirs
from skorch.classifier import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sleepwalker.trainer.utils import cohen_kappa_from_confusion_matrix

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
# N_VAL_SAMPLES=int(os.environ.get('N_VAL_SAMPLES', 500))
# N_TEST_SAMPLES=int(os.environ.get('N_TEST_SAMPLES', 500))
N_WORKERS_DATASET=int(os.environ.get('N_WORKERS_DATASET', 24))
N_WORKERS_DATALOADER=int(os.environ.get('N_WORKERS_DATALOADER', 24))
DEVICE=os.environ.get('DEVICE', 'cuda')
#DEVICE='cpu'

def prepare_sleep_staging_patient(data_df, label_df, label_extra_df, **_kwargs):
    trimmed = trim_event(data_df, label_df, label_extra_df)
    if trimmed is None:
        return None
    # has_airflow = 'Nasal Pressure' in data_df.columns or 'Airflow' in data_df.columns
    # if not has_airflow:
    #     return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df

def get_collate_ignore_list(dataset) -> list[str]:
    return ['time', 'dataset', 'target', 'data'] if isinstance(dataset, MultiDataset) else ['time', 'target', 'data']

def prepare_multiclass_sample(data, target, task_config, **item):
    new_item = {}
    new_item['patient'] = torch.tensor(int(hashlib.sha256(item['patient'].encode('utf-8')).hexdigest(), 16) % 10**8).long()

    # Build individual data modalities
    for modality_name in dict.fromkeys(data.columns):  
        modality_df = data.loc[:, data.columns == modality_name]
        new_item[f'data_{modality_name}'] = torch.from_numpy(modality_df.values).float()

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
    #all_patients = all_patients[:50]
    train_patients, rest = random_split(all_patients, test_frac=0.33, seed=1912817)
    val_patients, test_patients = random_split(rest, test_frac=0.5, seed=918171)

    channels = [
        ChannelConfig('EEG', ['C4-M1', 'F4-M1', 'O2-M1', 'C3-M2', 'F3-M2', 'O1-M2'], EEGFilterNormalizer(fs=SAMPLE_FREQUENCY)),
        ChannelConfig('EOG', ['E1-M2', 'E2-M1'], EEGFilterNormalizer(fs=SAMPLE_FREQUENCY)),
        ChannelConfig('RIP', ['RIP Flow', 'RIP Sum', 'Abdomen', 'Chest'], SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=10.0, band_order=4, notch_freq=None)),
        ChannelConfig('SPO2', ['Saturation'], SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=10.0, band_order=4, notch_freq=None)),
        ChannelConfig('LEG-EMG', ['Left Leg', 'Right Leg'], SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=10.0, highcut=45.0, band_order=4, notch_freq=None)),
        ChannelConfig('ECG', ['ECG'], SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, highcut=45)),
    ]
    event_mapping = {
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

    logger.context('TRAIN')
    train_ds = Ruhrlandklinik(
        channels=channels, 
        event_mapping=event_mapping,
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
        channels=channels, 
        event_mapping=event_mapping,
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
        channels=channels, 
        event_mapping=event_mapping,
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
    #val_sampler = torch.utils.data.RandomSampler(val_ds, num_samples=min(len(val_ds), N_VAL_SAMPLES)) 
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        #sampler=val_sampler,
        num_workers=N_WORKERS_DATALOADER, 
        collate_fn=partial(batch_collate, ignore_list=get_collate_ignore_list(val_ds)), 
        drop_last=False, 
        pin_memory=True
    )
    #test_sampler = torch.utils.data.RandomSampler(test_ds, num_samples=min(len(test_ds), N_TEST_SAMPLES)) 
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        #sampler=test_sampler,
        num_workers=N_WORKERS_DATALOADER, 
        collate_fn=partial(batch_collate, ignore_list=get_collate_ignore_list(test_ds)), 
        drop_last=False, 
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

from sleepwalker.models.utils import AttentionPooling

class AttentionClassifier(nn.Module):

    def __init__(self, n_classes, embedding_dim):
        super().__init__()
        self.channel_pool = AttentionPooling(dim=embedding_dim, attn_size=embedding_dim, along_dimension=2)
        self.time_pool = AttentionPooling(dim=embedding_dim, attn_size=embedding_dim, along_dimension=1)
        self.prediction_head = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        x = rearrange(x, 'B N F D -> B N D F')
        x, _ = self.channel_pool(x) # out: (B, N, F)
        x, _ = self.time_pool(x) # out: (B, F)
        x = torch.nn.functional.softmax(self.prediction_head(x), dim=-1)
        return x

class AttentionClassifierTrainer:
    """One trainer per task. Wraps a frozen embedding model + a trainable
    AttentionClassifier head.

    `fit` estimates per-patient means over BOTH train_dl and val_dl up front
    (their patients are disjoint), then trains the head on the frozen,
    per-patient-centered embeddings for N epochs with early stopping on val_dl.
    For test-set evaluation, call `estimate_means(test_dl)` once before
    `predict`/`evaluate`.
    """

    def __init__(self, base_model, task_name, task_config, device,
                 lr=1e-3, max_epochs=100, patience=10):
        self.base_model = base_model
        self.task_name = task_name
        self.cfg = task_config[task_name]
        self.device = device
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience

        # modalities this task actually needs
        self.modalities = self.cfg['embeddings']
        self.n_slices = self.cfg['n_slices']
        self.n_classes = len(self.cfg['labels'])

        # built lazily in fit (need embedding_dim from a real batch)
        self.clf = None
        self.optimizer = None
        self.loss_fn = self.cfg['loss_function']
        self.class_weights = self._build_class_weights()

        # per-patient means, keyed by patient id (shared across all loaders;
        # ids do not collide across splits)
        self._means = {m: {} for m in self.modalities}

    # ---------------------------------------------------------------- helpers
    def _build_class_weights(self):
        cw = self.cfg.get('class_weights')
        if cw is None:
            return None
        w = torch.tensor([cw[i] for i in range(self.n_classes)],
                         dtype=torch.float, device=self.device)
        if self.cfg.get('loss_mode') == 'inverse':
            w = 1.0 / w
        return w

    def _embed(self, batch):
        """Run the frozen base model for this task's modalities only."""
        emb = {}
        for m in self.modalities:
            _x = batch[f'data_{m}'].to(self.device)
            _x = self.base_model.apply_preprocessors(_x, m)
            emb[m] = self.base_model.encode(_x, m)
        return emb, batch['patient']

    def _assemble(self, emb):
        """Concat the task's modalities, fix dims, slice out the middle."""
        X = torch.cat([emb[m] for m in self.modalities], dim=-1)
        if X.dim() < 4:
            X = X.unsqueeze(-1)
        T = X.shape[1]
        n = self.n_slices
        return X[:, T // 2 - n // 2 : T // 2 + n // 2]

    # -------------------------------------------------------- patient centering
    def estimate_means(self, dl):
        """Accumulate per-patient sums over `dl` and merge into stored means.
        Call once per loader (train, val, test) before training/eval."""
        self.base_model.eval()
        sums = {m: {} for m in self.modalities}
        counts = {m: defaultdict(int) for m in self.modalities}
        with torch.no_grad():
            for batch in dl:
                emb, pids = self._embed(batch)
                for m in self.modalities:
                    e = emb[m]
                    for uid in torch.unique(pids):
                        uid_i = int(uid)
                        mask = pids == uid
                        s = e[mask].sum(dim=0)
                        if uid_i not in sums[m]:
                            sums[m][uid_i] = s.clone()
                        else:
                            sums[m][uid_i] += s
                        counts[m][uid_i] += int(mask.sum())
        for m in self.modalities:
            for uid in sums[m]:
                self._means[m][uid] = sums[m][uid] / counts[m][uid]
        return self

    def _center(self, emb, pids):
        """Center using stored per-patient means (estimated up front for
        every loader). Errors loudly if a patient was never estimated."""
        out = {}
        for m in self.modalities:
            e = emb[m].clone()
            means = self._means[m]
            for uid in torch.unique(pids):
                uid_i = int(uid)
                mask = pids == uid
                if uid_i not in means:
                    raise KeyError(
                        f"No mean for patient {uid_i} (modality {m}); "
                        f"call estimate_means on this loader first.")
                e[mask] -= means[uid_i].to(e.device)
            out[m] = e
        return out

    def _build_head(self, sample_emb):
        X = self._assemble(sample_emb)
        embedding_dim = X.shape[2]
        self.clf = AttentionClassifier(
            n_classes=self.n_classes, embedding_dim=embedding_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.lr)

    # ----------------------------------------------------------------- fit
    def fit(self, train_dl, val_dl):
        self.base_model.eval()

        # Estimate per-patient means for every loader we will consume.
        self.estimate_means(train_dl)
        self.estimate_means(val_dl)

        # Build the head from a sample batch.
        with torch.no_grad():
            sample_emb, _ = self._embed(next(iter(train_dl)))
        self._build_head(sample_emb)

        best_kappa = -np.inf
        best_state = None
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            # ---- train ----
            self.clf.train()
            for batch in tqdm.tqdm(train_dl, total=len(train_dl), desc=f'[{self.task_name} | Train | Epoch={epoch}]', disable=True):
                with torch.no_grad():
                    emb, pids = self._embed(batch)
                    emb = self._center(emb, pids)
                    X = self._assemble(emb)
                y = batch[f'target_{self.task_name}'].argmax(-1).to(self.device)

                self.optimizer.zero_grad()
                logits = self.clf(X)
                loss = self.loss_fn(logits, y, weight=self.class_weights)
                loss.backward()
                self.optimizer.step()

            # ---- validate / early stop ----
            train_kappa = self.evaluate(train_dl)
            val_kappa = self.evaluate(val_dl)
            print(f'[{self.task_name} | Epoch {epoch}] Train Kappa: {train_kappa:.3f} | Val. Kappa: {val_kappa:.3f}')
            if val_kappa > best_kappa:
                best_kappa = val_kappa
                best_state = {k: v.detach().clone()
                              for k, v in self.clf.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.clf.load_state_dict(best_state)
        self.best_kappa_ = best_kappa
        return self

    # ----------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, dl):
        self.clf.eval()
        preds, trues = [], []
        for batch in dl:
            emb, pids = self._embed(batch)
            emb = self._center(emb, pids)
            X = self._assemble(emb)
            logits = self.clf(X)
            preds.append(logits.argmax(-1).cpu())
            trues.append(batch[f'target_{self.task_name}'].argmax(-1).cpu())
        cm = confusion_matrix(torch.cat(trues).numpy(), torch.cat(preds).numpy())
        return cohen_kappa_from_confusion_matrix(cm)

    @torch.no_grad()
    def predict(self, dl):
        self.clf.eval()
        preds = []
        for batch in dl:
            emb, pids = self._embed(batch)
            emb = self._center(emb, pids)
            X = self._assemble(emb)
            preds.append(self.clf(X).argmax(-1).cpu())
        return torch.cat(preds).numpy()

def main():
    train_ds, val_ds, test_ds = get_datasets()
    train_dl, val_dl, test_dl = get_dataloader(train_ds, val_ds, test_ds)

    groups = val_ds.channel_groups

    model = MaskedAutoencoder(
        groups=groups,
    )
    # TODO: For new cohort, it might be a good idea to redo the warmup to normalize
    #       For now (and for simplicity) we load the normalization weights as the cohort is the same
    n_steps = model.window_size // 2 + 1
    for modality, n_channels in {k: len(v) for k, v in groups.items()}.items():
        model.preprocessors[modality][1].mean = torch.zeros(n_steps, n_channels)
        model.preprocessors[modality][1].M2 = torch.zeros(n_steps, n_channels)
        model.preprocessors[modality][1].count = torch.zeros(())

    checkpoint = torch.load('checkpoint.pt', map_location=DEVICE)

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    trainers = {}
    for task in ['breathing', 'legmovement', 'sleep']:
        t = AttentionClassifierTrainer(
            base_model=model, task_name=task,
            task_config=task_config, device=DEVICE,
            max_epochs=10,
        )
        t.fit(val_dl, test_dl)        # your "val_dl" is the downstream train set
        print(task, t.best_kappa_)
        print(' ')

        # later, on a true held-out test loader:
        # t.estimate_means(heldout_dl)
        # print(task, t.evaluate(heldout_dl))
        trainers[task] = t


    # print('Number of parameters:', count_parameters(model))
    # model_hp = model.get_hyperparameters()

    # print('Tracking to MLFLOW instance', os.environ['MLFLOW_URL'])
    # logger.add_sink(MlflowSink(tracking_uri=os.environ['MLFLOW_URL'], experiment='debug', artifact_uri=None))
    # logger.start_run(run_name='testrun-full', params=model_hp)

    # EPOCHS=31
    # trainer = MaskedAutoencoderTrainer(
    #     groups=train_ds.channel_groups,
    #     epochs=EPOCHS,
    #     optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=2e-3),
    #     #lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=EPOCHS),
    #     classes=train_ds.get_classes(),
    #     downstream_tasks=normalized_task_config,
    #     save_every=10,
    #     device=DEVICE,
    #     warmup_device='cpu',
    # )

    # trainer.fit(model, train_loader=train_dl, val_loader=val_dl, test_loader=test_dl)

if __name__ == '__main__':
    main()
