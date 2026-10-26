# Model packages and isolated inference

The API supports the new `sleepwalker-packaged-model-v1` directories:

```text
models/
  sw-sleep/
    manifest.json
    model.pt
  sw-arousal/
    manifest.json
    model.pt
  ...
```

No `models.json`, additional API descriptor, repacking or `prepare_model` step
is needed. Old `.swmodel` archives and historical RUTime adapters are no longer
part of the server's inference path. Original package files are never rewritten.

## Separate framework environment

The provided models README pins the compatible framework revision to
`2c77bf61671a2d3690d6a29d73fd5eece7e1c544`. The manifests' `git_commit`
identifies the models repository, not that framework revision.

Keep this framework in its own environment. Do not upgrade the current API
checkout or install the new framework into the API's venv:

```bash
python3 -m venv /path/to/inference-venv
/path/to/inference-venv/bin/python -m pip install --upgrade pip
/path/to/inference-venv/bin/python -m pip install -r sleepwalker/server/requirements-inference.txt
```

Replace the example paths with persistent locations. Start from the API project
directory using its existing venv:

```bash
python -m sleepwalker.server \
  --models sleepwalker/server/models \
  --inference-python /path/to/inference-venv/bin/python \
  --device cpu --batch-size 1
```

There is no environment-variable configuration and no implicit Python fallback.
Starting without `--models` still runs the API without inference.

The persistent local inference Python is
`/cephfs/users/oleksjuk/.venv/sleepwalker_inference_venv/bin/python`.
It has its own installed framework and dependencies, without shared package paths
from the API environment or runtime imports from `/tmp`. The requirements file
pins the framework revision and the tested direct dependency versions, including
CPU-only Torch 2.13.0. It is not an exhaustive lock of all transitive dependencies.
For CUDA, choose and test an appropriate Torch build separately.

Keep the API venv activated and pass the separate Python explicitly:

```bash
python -m sleepwalker.server \
  --models sleepwalker/server/models \
  --inference-python /cephfs/users/oleksjuk/.venv/sleepwalker_inference_venv/bin/python \
  --device cpu --batch-size 1
```

No server restart is performed automatically when setting up the environment.
If the server still uses the old `/tmp` environment, restart it manually with
the command above; this server's in-memory sessions/jobs do not survive a restart.

## Registry and input contracts

The API registry reads JSON only, not pickled model objects or weight tensors.
It scans immediate subdirectories for `manifest.json`.

| API field | Manifest source |
| --- | --- |
| id, name | name |
| task | task, unchanged (for example `sleep`, not an inferred alias) |
| classes | classification.classes, preserving order |
| input_channels | input_channels (logical names) |
| accepted EDF names | config.data.channels[].physical_names |
| output_resolution | classification.target_resolution / classification.sequence_len |
| output_kind | epoch for the supported single-head fixed-interval contract |

The supplied expert IDs are `sleep-hsp-development`, `arousal-hsp-development`,
`breathing-hsp-development` and `desaturation-hsp-development`; their folder names
are not their API IDs. The four temporal steps are 30, 1, 5 and 10 seconds.

A request prechecks declared physical aliases, e.g. `C3-M2` instead of the
logical name `eeg`. The official package performs authoritative channel/unit
validation, resampling, filtering and normalization. Pass the original EDF;
the API does not recreate training configurations or preprocess signals twice.
The small `tests/data/signals_01.edf` does not contain all these packages'
required channel names/signals and is not a suitable real-model inference demo.

Only `single-head-multiclass` packages are enabled. Multi-task packages such as
`joint` are explicitly skipped with a warning. They are not advertised as
available models until overlap aggregation and multi-task result handling are
implemented. Broken single-task packages stop registry initialization instead of
being silently hidden.

## Runtime and results

`PackageInferenceRunner` starts `inference_worker.py` as an absolute script
using the configured Python. Its working directory is a private temporary
directory; inherited `PYTHONPATH`/`PYTHONHOME` are removed. Inherited
`LD_LIBRARY_PATH`/`LD_PRELOAD` are also removed for this worker only, preventing
cluster/container overrides from loading system libtorch binaries instead of
the libraries bundled with the inference environment. This prevents both the
older API checkout from shadowing the new framework's `sleepwalker` package
and accidental mixing of Python and native Torch versions. The API process
and the user's shell environment are not modified.

The worker uses the official `load_packaged_model()` and `predict_dataset()`.
It follows `predict_patient()`'s public preparation steps (clone the saved
dataset, validate compatibility and initialize one EDF with strict checking).
The explicit dataset call lets us own the official `EDFCache` under the worker's
private temporary directory instead of leaving cache files in `/dev/shm` after
forced termination. No framework code or preprocessing algorithms are patched.
The official loader checks the payload SHA256. The worker also checks the
executable task, ordered classes, logical inputs and output resolution against
the API metadata. Packages are trusted administrator-installed Python objects:
checksums detect accidental changes but do not make untrusted pickle files safe.
Do not use user uploads as model packages.

The original EDF stays on disk. Only request metadata and prediction tables cross
the process boundary, via private JSON/CSV files, never full signal arrays.
Framework timestamps mark interval starts. The runner converts them once to the
API mapper's existing center convention; e.g. start 300 s, resolution 30 s becomes
center 315 s internally and API interval [300, 330]. Existing exports/imports can
therefore retain their interval semantics.

The runner rejects empty/invalid output, wrong tasks, bad probabilities,
duplicate timestamps and out-of-recording intervals. All selected models must
succeed before results are published; failure/cancellation publishes no partial
batch. There is no guessed padding at recording edges or guessed event
segmentation from class names.

The default batch size is 1 to keep peak memory conservative; it can be increased
with `--batch-size` after checking available resources. Each model uses a fresh
process with two CPU Torch threads, no dataset/loader
workers and one view per window. The official EDF cache can still start its own
multiprocessing manager, so local IPC sockets must be allowed. Job cancellation
stops the worker's process group, including cache children. The API parent
removes the transport directory and its recording cache even after SIGKILL.
Worker termination also releases its model/signal memory; this is not a guarantee about peak memory
for a full-night recording. Temporary transport files are removed on success,
failure and cancellation. Worker tracebacks go to server logs, not public errors.

If the worker catches Python `MemoryError` or Torch `OutOfMemoryError`, the job
fails with `RESOURCE_EXHAUSTED` and a fixed public message, available through both
job status and SSE updates. Allocator details stay in server logs. An abrupt
worker exit without a valid response remains `COMPUTE_UNAVAILABLE`: SIGKILL alone
does not prove memory exhaustion. This reports detectable memory failures; it
does not impose memory limits, reduce peak memory or automatically retry jobs.

Progress phases currently describe starting the worker and handing preprocessing/
inference to the package; they are not per-window percentage measurements.

## Tests

The normal server suite uses metadata fixtures and tiny subprocess workers, and
does not require private weights, patient data or the new framework installed
in the API environment. The opt-in test exercises the real path end to end
with synthetic 690-second EDFs:

```bash
pytest -q tests/test_server_packaged_models.py \
  --inference-python /path/to/inference-venv/bin/python \
  --inference-models sleepwalker/server/models
```

It covers all four experts: upload, alias resolution, job completion, prediction
retrieval, exact interval boundaries and session deletion. Additional cases check
payload checksums, executable/manifest disagreement, insufficient context and
cache cleanup on cancellation. Without those options the eight tests are skipped.
These are technical compatibility tests, not measures
of predictive quality or production load/memory behavior.
