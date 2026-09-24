# Roadmap

This page records useful features discovered while writing and testing the documentation. Entries describe missing or insufficiently tested behavior; they are not part of the supported API yet.

## Publish pretrained model packages {#publish-pretrained-model-packages}

**Current state:** Sleepwalker does not publish ready-to-use sleep-staging packages and has no model registry. Users must train a package or receive one from a trusted collaborator.

**Target:** Publish versioned packages with checksums, model cards, training data and split information, expected EDF channels and units, validation metrics, license information, and a tested loading example.

## Add a public Sleep-EDFx download API {#add-a-public-sleep-edfx-download-api}

**Current state:** `SleepEDFx.py` contains script-level download code, but there is no supported importable function with a documented return value and tests. The documentation therefore directs users to PhysioNet's download methods.

**Target:** Provide a public function that selects Sleep Cassette and/or Sleep Telemetry, creates a predictable directory structure, resumes partial downloads safely, verifies checksums, and returns the dataset paths. Add unit tests for existing files, corrupt files, partial downloads, and HTTP errors.

## Stabilize command-line prediction and evaluation {#stabilize-command-line-prediction-and-evaluation}

**Current state:** `PackagedModel.predict_patient()` is the documented prediction path. There is no `sleepwalker predict` command. `tools/predict.py` is marked as a legacy table transformation. `sleepwalker evaluate` and `sleepwalker evaluate-system` exist, but their intended public scope and end-to-end coverage still need review before they are presented as the standard route for users.

**Target:** Add a prediction command that loads a package, accepts one EDF or a manifest, and writes the same table as `predict_patient()`. Decide which evaluation commands are public, give them stable configuration examples, and cover package loading, EDF preparation, output files, failures, and CPU execution with end-to-end tests.

## Add patient-level embedding methods {#add-patient-level-embedding-methods}

**Current state:** Surfaced while writing the [Extract embeddings](how-to/extract-embeddings.md) guide: there is no first-class embedding API, so the guide has to hand-roll an `embed_patient()` helper that clones the packaged dataset, builds an ordered loader, and calls `model.features()` per window. `PackagedModel` exposes `predict_patient()` but no `embed_patient()` or `embed_dataset()`, so the embedding path is asymmetric with prediction and every user must reimplement the plumbing.

**Target:** Add methods parallel to `predict_patient()` and `predict_dataset()` with defined timestamp columns, feature naming, repeated-view behavior, and tests for embedding-only and classifier-plus-embedding packages, so the guide can call `package.embed_patient(...)` instead of defining its own helper.

## Make repeated-view inference reproducible {#make-repeated-view-inference-reproducible}

**Current state:** The prediction `seed` initializes the data loader's PyTorch generator. With `num_workers_loader=0`, random operations inside custom sample-preparation callbacks may use Python, NumPy, or global PyTorch state that is not reset by this argument. Reproducibility has not been established for every worker configuration.

**Target:** Define which random generators callbacks may use, seed them consistently for each patient and view, and test identical results across repeated runs and supported worker counts.

## Define deterministic EDF file discovery {#define-deterministic-edf-file-discovery}

**Current state:** `get_edf_files_in_repo()` returns relative or absolute strings according to the supplied root and preserves filesystem traversal order. Callers must sort results themselves when order matters.

**Target:** Decide whether the public function should return resolved `Path` objects or strings, guarantee deterministic ordering, and test recursive and non-recursive discovery.

## Expand generated API coverage {#expand-generated-api-coverage}

**Current state:** The [Python API](reference/api.md) is rendered from signatures, type annotations, and docstrings with `mkdocstrings`. The list and grouping of documented objects is curated in `api.md` and currently covers the objects used by the guides.

**Target:** Decide which modules form the public API, expand the generated coverage to those modules, and fail the documentation build when a documented public import or cross-reference no longer resolves.

## Document multitask and overlapping-event targets {#document-multitask-targets}

**Current state:** The [multilabel target guide](how-to/multilabel.md) now documents the task configuration, the padded target and mask tensor shapes, `prepare_multitask_target()`, and `MultiLabelTrainer`, and the trainer and target helpers are rendered in the [trainers reference](reference/trainers.md). What is still missing is a verified end-to-end example: no YAML in `configs/` trains `MultiLabelTrainer`, and the only multitask-shaped model in the library is `ModelGraphClassifier`, so a simple two-head `BaseModel` example has not been run against real data.

**Target:** Add one working configuration with overlapping annotations (dataset + two-head model + `MultiLabelTrainer`), verify it end to end, and link it from the guide.

## Document and validate channel normalizers {#document-and-validate-channel-normalizers}

**Current state:** Only `EEGFilterNormalizer` appears in the [Python API](reference/api.md). The `Normalizer` protocol and `SignalFilterNormalizer`, which implements the Butterworth band-pass, notch, and `(x - mean) / std` rescaling steps, have no generated reference entry and no docstrings, and the other normalizers (`PulseFilterNormalizer`, `RespirationFilterNormalizer`, `SaturationFilterNormalizer`) are undocumented. Nothing checks that the `fs` given to a normalizer matches the dataset `sample_frequency`, so a mismatch silently filters at the wrong frequencies.

**Target:** Document the normalizer interface, each built-in normalizer with its default filter settings, and the order in which normalizers run relative to unit conversion, resampling, rereferencing, and z-normalization. Add docstrings so the reference renders, and reject or warn about a normalizer frequency that disagrees with `sample_frequency`.

## Document target interval placement {#document-target-interval-placement}

**Current state:** The annotation window handed to `prepare_target` spans the whole input window and `time` is the window start. Which sub-interval is supervised is chosen inside the callback through `target_resolution`, `target_position`, and `target_offset`; `BaseDataset` exposes no placement option, and the deployment contract does not record one.

**Target:** Decide whether temporal placement belongs to the dataset configuration, the model/expert contract, or the callback, document the resulting invariant in one place, and state how packaged models describe the position of their outputs relative to the input window.

## Re-export dataset building blocks {#re-export-dataset-building-blocks}

**Current state:** `ChannelConfig` and `batch_collate` live in `sleepwalker.datasets.Basedataset` and are not re-exported, so `from sleepwalker.datasets import ChannelConfig` raises `ImportError`. The [loading-data guide](how-to/data.md) therefore imports them from `sleepwalker.datasets.Basedataset`; other guides still use the shorter form and currently fail.

**Target:** Decide the public import surface of `sleepwalker.datasets`, re-export the objects that every example needs, list them in `__all__`, and cover the documented imports in a test that imports every object named in the guides.

## Implement the DCSM dataset adapter {#implement-the-dcsm-dataset-adapter}

**Current state:** `sleepwalker.datasets.DCSM` provides download helpers and a SHA256 table for the DCSM-Demo files, but no `DCSM` adapter class. The signals are stored as HDF5 (`psg.h5`) and the hypnogram as a `hypnogram.ids` file, neither of which fits the EDF-based reader used by every other adapter, so no dataset can be constructed from the module and the [reference page](reference/datasets/DCSM.md) documents only the download helpers.

**Target:** Decide whether DCSM is supported at all. If it is, add an adapter that reads the HDF5 signals and the `hypnogram.ids` annotations, maps them to `ChannelConfig` logical channels and units, and pass the same initialization and windowing tests the EDF adapters use. Otherwise remove the module from the documented dataset list.

## Make batch balancing work in DataLoader workers {#make-batch-balancing-work-in-dataloader-workers}

**Current state:** `MulticlassTrainer` implements `balance_batches` by wrapping the dataset `prepare_target` callback during warmup, after class counts have been estimated. `build_loader` uses `persistent_workers=True` whenever `num_workers > 0`, so the workers keep their own dataset copy with the unwrapped callback and the balancing is silently skipped; the code carries a matching `TODO`. Batch balancing also requires a live EDF-backed dataset and is rejected for numpy caches.

**Target:** Implement balancing where it takes effect, either by rebuilding the loader after count estimation or by moving acceptance into a sampler, cover it with a test that uses `num_workers > 0`, and report the achieved class distribution so a silently disabled option becomes visible.

## Separate test evaluation from the training runner {#separate-test-evaluation-from-the-training-runner}

**Current state:** `run()` trains, exports, and then evaluates every dataset in `test_datasets` inside the same call, writing the combined record to `results.json`. The module docstring marks the testing step as an open question. There is no runner entry point that trains and exports without an evaluation set, and the `dry` path inherits the same structure.

**Target:** Split training/export from test evaluation so a run can be trained without test data, keep the single-call convenience for scripts that want it, and document which artifact holds training-only results.

## Detect the training device instead of defaulting to a GPU {#detect-the-training-device}

**Current state:** `MulticlassTrainer` and the other trainers default to `device="cuda:0"`. Only the `dry` command rewrites that to `cuda:0 if torch.cuda.is_available() else cpu`, so `sleepwalker train` on a CPU-only machine initializes the datasets, builds the model, prints the summary, and then fails inside `torch.cuda._lazy_init` unless every trainer section sets `device: cpu` by hand.

**Target:** Resolve the device once in configuration or in the runner, warn when a requested CUDA device is unavailable, and let the examples run unchanged on a CPU-only machine.

## Resume a run without discarding completed work {#resume-runs-reliably}

**Current state:** `sleepwalker resume <checkpoint>` restores trainer state and rebuilds the datasets from `hparams.yml`, then calls the same `run()`, which raises `FileExistsError` when `<run>/results.json` already exists. A finished run therefore cannot be resumed, not even to train more epochs, and with the default `save_every=10` a short run that dies inside the first nine epochs leaves no intermediate checkpoint at all.

**Target:** Make continuation explicit: allow a resumed run to append to or version its result file, report the epoch range that will actually run before starting, and document which checkpoints exist for a run of a given length.

## Accept optimizer factories that receive the model {#accept-optimizer-factories-that-receive-the-model}

**Current state:** `BaseTrainer.fit()` calls the configured optimizer factory with the model module: `self.optimizer_fn(model)`. Torch optimizers expect an iterable of parameters, so hand-written Python must use `lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)`; the natural-looking `partial(torch.optim.Adam, lr=1e-3)` fails at the first `fit()` with `TypeError: 'Linear' object is not iterable`. The YAML path injects `model.parameters()` itself via `config.build_factory`, so only direct Python construction is affected. The [sleep-staging tutorial](how-to/train-sleep-staging.md) and the [multiclass guide](how-to/train-multiclass.md) now show the lambda form.

**Target:** Accept both forms — recognize an `nn.Module` argument inside the optimizer construction and substitute `model.parameters()` — or normalize optimizer factories during configuration. Cover both spellings with a trainer test and state the accepted factory signature in the API reference.

## Expose trainer predictions {#expose-trainer-predictions}

**Current state:** `BaseTrainer.fit()` and `BaseTrainer.test()` return scalar losses and accumulated confusion matrices only; no trainer implements a `predict` method, yet the [Python API](reference/api.md) lists `predict` as a member of `MulticlassTrainer`, where it renders nothing. Getting the predicted label of every window requires re-running the model over a loader manually or going through the exported `PackagedModel`.

**Target:** Decide whether trainers should return (or log) per-window predictions — patient, time, predicted class, and per-class probabilities — from `fit()` and `test()`, or remove the `predict` entry from `api.md` and point users at the packaged prediction path instead. Implement the chosen direction and cover it with a test.

## Reuse open EDF reader handles {#reuse-open-edf-reader-handles}

**Current state:** `EDFFile` declares a `handle` field, but `EDFFile.get_x()` passes the file path to `edf_to_df()`, which opens a new `pyedflib.EdfReader` for every requested window and closes it afterwards. Every sample access therefore pays a file open, header parse, and close on top of the window read itself.

**Target:** Keep one open reader per patient per worker (opened lazily on first window access, closed when the worker exits or the dataset is re-initialized), verify correctness under `num_workers > 0` and `persistent_workers=True`, and measure the per-window speedup so the docs can state it.

## Provide a supported thread-limit helper {#provide-a-supported-thread-limit-helper}

**Current state:** The performance guide tells users to set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` before importing NumPy to avoid BLAS oversubscription inside DataLoader workers. The library itself provides no helper; the legacy research scripts set the variables ad hoc at module import, and forgetting the import-order requirement silently leaves every worker with one thread per core.

**Target:** Add a documented utility (e.g. `sleepwalker.utils.set_thread_limits(threads_per_worker=2)`) that sets the relevant environment variables and PyTorch thread settings, fails with a clear message if called after the numeric libraries are already imported, and is used by the runner and the guides.

## Support patient-balanced sampling for numpy caches {#support-patient-balanced-sampling-for-numpy-caches}

**Current state:** `PatientSampler` requires `get_patient_ranges()`, which `NumpyDataset` does not implement, so `build_loader(..., sampling="patient_balanced")` raises a `ValueError` on a cached dataset. Trainer-side batch balancing is likewise rejected for numpy caches. The exported `patient.npy` array already identifies the per-patient blocks needed to build the ranges.

**Target:** Implement `get_patient_ranges()` on `NumpyDataset` from the cached patient identifiers, decide whether batch balancing should also work on caches, and cover both with tests so the EDF and cache paths share one sampling contract.

## Fix dangling documentation links {#fix-dangling-documentation-links}

**Current state:** Several guides link to pages that do not exist yet: `how-to/dataset-windows.md` is referenced from `getting-started/index.md`, `how-to/index.md`, and `how-to/sleep-edfx.md`, while the existing `how-to/data.md` carries the window-reading content. The documentation build is not checked in CI, so such links are not reported.

**Target:** Either add the missing pages or point the links at the existing pages, keep the navigation in `zensical.toml` complete, and fail the documentation build on unresolved internal links.
