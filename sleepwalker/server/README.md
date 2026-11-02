# Sleepwalker API

Runs Sleepwalker models on sleep recordings and lets a client view signals and
predictions, correct them and export the result:

```text
upload EDF -> run inference (job) -> view signals + predictions -> correct -> export -> delete
```

Everything is temporary by design: sessions live in memory and on disk under a
temporary directory, and expire after 2 hours without access. A session survives
a restart only if it was exported as a workspace and imported again.

## Start

The API and the models use **separate Python environments**, because the models
need a newer framework than this checkout. See [Inference](#inference).

```bash
python3 -m venv /path/to/inference-venv
/path/to/inference-venv/bin/python -m pip install -r sleepwalker/server/requirements-inference.txt

python -m sleepwalker.server \
  --models sleepwalker/server/models \
  --inference-python /path/to/inference-venv/bin/python \
  --device cuda --batch-size 1
```

The inference requirements install a CUDA 13.0 build of Torch. Without a GPU, use
`--device cpu`; that is also the default.

Without `--models` the server runs without inference. It listens on
`127.0.0.1:6006` and has no authentication, so do not expose it to a network.
Interactive API docs are at `/docs`. The test UI lives in [`ui/`](../../ui/README.md).

## Endpoints

All paths start with `/api/v1`, except `/health`.

| Method and path | Purpose | Response |
| --- | --- | --- |
| `GET /health` | Liveness check | 200 |
| `GET /models` | List available models | 200 |
| `GET /models/{id}` | One model | 200 |
| `POST /sessions` | Upload a study (`files`, multipart) | 201 session |
| `GET /sessions/{id}` | Session state, signals, tasks | 200 |
| `DELETE /sessions/{id}` | Cancel its jobs and delete all its data | 204 |
| `POST /sessions/{id}/inference` | Start models: `{"models": ["id"]}` | 202 job |
| `GET /sessions/{id}/tasks` | Tasks that have predictions | 200 |
| `GET /sessions/{id}/predictions/{task}?start=&end=` | Predictions in a time range | 200 |
| `GET /sessions/{id}/signals` | List signals | 200 |
| `GET /sessions/{id}/signals/{signal}?start=&end=&max_points=` | Reduced signal data | 200 |
| `GET /sessions/{id}/annotations` | Edit history and current corrections | 200 |
| `POST /sessions/{id}/annotations` | `set_label`, `add_event`, `delete_event`, `revert` | 201 |
| `POST /sessions/{id}/exports` | Start an export | 202 job |
| `GET /exports/{id}` | Export state | 200 |
| `GET /exports/{id}/download` | Download a finished export | 200 file |
| `POST /sessions/import` | Upload a workspace (`bundle`, multipart) | 202 job |
| `GET /jobs/{id}` | Job state | 200 |
| `GET /jobs/{id}/events` | Live job updates (SSE) | 200 stream |
| `POST /jobs/{id}/cancel` | Request cancellation | 202 |

Times are seconds since the start of the recording. Errors always look like this:

```json
{"error": {"code": "SESSION_NOT_FOUND", "message": "Session not found.", "details": null}}
```

## Code map

`api/` only translates between HTTP and the other folders.

```text
server/
  __main__.py   command line entry point
  app.py        create_app: builds all objects and connects them
  models/       trained model packages
  api/          everything about HTTP
  core/         what every session needs
  recording/    the uploaded study
  inference/    finding and running models
  results/      predictions and what is done with them
```

| Module | Contains |
| --- | --- |
| `app.py` | `create_app` wiring, startup and periodic cleanup |
| `api/*.py` | One router per topic |
| `api/deps.py` | Dependencies for routes, e.g. `SessionDep` (404 if unknown) |
| `api/errors.py` | `APIError`, its helpers and the error format |
| `api/schemas.py` | All request and response models |
| `core/sessions.py` | `Session`, TTL handling |
| `core/jobs.py` | Job states, `JobExecutor`, SSE mailbox |
| `core/storage.py` | Temporary files on disk |
| `core/lifecycle.py` | Deletes everything belonging to a session |
| `recording/study.py` | Recognise and validate uploaded files |
| `recording/signals.py` | Bounded EDF range reads for REST |
| `recording/plots.py` | `plotly-resampler` figures for the Dash UI |
| `inference/catalog.py` | `ModelCatalog`: which models exist right now |
| `inference/runner.py` | Starts the worker, checks and stores its results |
| `inference/worker.py` | Runs **in the inference venv**, loads the model |
| `results/predictions.py` | `PredictionStore`, conversion to API records |
| `results/annotations.py` | Corrections as an append-only edit list |
| `results/exports.py` | JSON result export and workspace contents |
| `results/workspace.py` | Workspace format, safety checks, import |

To add an endpoint, add a route to the matching `api/` module and its models to
`api/schemas.py`; the logic belongs in the topic's module.

## Models

Models are not registered. `ModelCatalog` reads the subfolders of `--models` on
every lookup, so a newly trained model is usable without a restart:

```text
models/
  sw-sleep/
    manifest.json   # read by the API
    model.pt        # only ever loaded by the worker
```

- A model's ID is `name` from its manifest. If several packages share a name, the
  one with the newest `manifest.json` wins.
- An invalid or half-copied package is skipped and logged once. To publish without
  a visible half-written state, copy to a temporary name and rename the folder last.
- A job fixes its packages when it starts. If a package is overwritten during the
  job, the job fails with a clear error and the next one uses the new package.
- The SHA-256 of the weights is stored as `version` in the session's provenance and
  in workspace exports, because the ID stays the same across retrainings.
- Multi-task packages (`joint`) are skipped until multi-task results are implemented.

The manifest fields used are `name`, `task`, `classification.classes`,
`input_channels`, `config.data.channels[].physical_names` (accepted EDF channel
names) and `classification.target_resolution / sequence_len` (output resolution).
Packages contain pickled Python objects: only install packages you trust, and never
accept them as uploads. Model packages are not part of the repository, so a fresh clone needs them copied into `models/`.

## Inference

`inference/runner.py` starts `inference/worker.py` as a separate process for each model,
using the configured Python. This has three purposes:

- **Compatibility:** the worker uses the framework version pinned in
  `requirements-inference.txt`. `PYTHONPATH`, `LD_LIBRARY_PATH` and similar variables
  are removed so that neither this checkout nor a system libtorch can interfere.
- **Cancellation:** cancelling kills the whole process group, including the
  framework's cache processes.
- **Memory:** all model memory is released when the worker exits.

Request and result travel as JSON and CSV through a private temporary directory
that is removed even after a forced kill. All selected models must succeed before
any result is published. One job runs at a time; others wait in `running`. The
worker's traceback goes to the server log, never to clients.

Prediction kinds are `epoch`, `continuous` and `event`. Today's packages produce
only `epoch`, and a manifest has no way yet to declare another kind; viewing,
correcting and exporting already handle all three.

## Jobs

Inference, export and import run in the background as jobs:
`queued -> running -> succeeded | failed`, and `cancelling -> cancelled`. A session
has at most one active job and does not expire while it runs. A failed job still
answers `200` on `GET /jobs/{id}`, with the reason in `error`.

`GET /jobs/{id}/events` sends named `job_update` events, ending with the final state,
and a `: keep-alive` comment every 15 seconds. Slow clients may miss intermediate
updates but never the final state. If the job is deleted, the stream sends
`job_deleted`. Disconnecting does not cancel the job. There is no replay: a
reconnecting client starts at the current state.

## Corrections

Corrections never change model output; they are appended as edits.
`GET .../annotations` returns `edits` (full history), `labels` (the latest active
`set_label` per task and start) and `events` (active human events).

- `set_label` applies to `epoch` tasks. `start` must be the start of an existing
  prediction interval, within 1 µs. It is not a multiple of the resolution: a
  desaturation interval `[35, 45]` is corrected with `start: 35`.
- `add_event` and `delete_event` apply to `continuous` and `event` tasks.
- `revert` undoes any edit except another `revert`.

## Signals

There are two independent ways to read signals:

- **REST** (`/signals/{signal}`) scans the range in chunks of 65,536 samples and
  returns the first and last sample plus the strongest value per bucket. `max_points`
  is 2–100,000. Memory stays bounded, but a whole night still takes disk and CPU time.
- **Plotly** (`app.state.plotly_service`, used by the Dash UI, no HTTP endpoint of its
  own) keeps the selected signals in RAM at full resolution, about 16 bytes per sample
  per view, until the view or session is closed. It must run in the same process as the API.

## Exports and workspaces

`POST .../exports` accepts two combinations:

- `{"format": "json", "mode": "results", "include": {...}}` writes a result document.
  `include` selects `predictions`, `probabilities`, `annotations` and `metadata`.
- `{"format": "sleepwalker", "mode": "workspace"}` writes `<id>.sleepwalker.zip`,
  which `POST /sessions/import` can restore:

```text
manifest.json           format version, task and model lists, SHA-256 of all files
study/metadata.json     recording metadata and signals
study/source/<file>     the original study files
predictions/0000.json   one document per task
annotations/edits.json  all edits in order
provenance/models.json  model metadata including version
```

Archives contain only data and are never executed. Checksums detect corruption but
do not prove authenticity. The upload is checked for unsafe paths, size limits
(10 GiB total, 256 MiB per JSON document, 10,000 files) and the format version. The
import job then verifies checksums, checks the study against the archive's metadata
and assigns new IDs to edits and events. A failed import leaves a `failed` session.

## Sessions and storage

A session's deadline moves forward on every access and when a job finishes. A
background task deletes expired sessions every 60 seconds. Files live under
`<tmp>/sleepwalker`; a lock file prevents two servers from sharing that directory,
and on startup leftover session folders from an earlier run are removed.

## Tests

```bash
pytest tests/test_server_*.py tests/test_job_*.py tests/test_catalog.py tests/test_export_service.py \
  tests/test_workspace_import.py ui/tests/test_integration.py  # main server tests

# Also runs the real models on synthetic EDFs:
pytest tests/test_server_packaged_models.py \
  --inference-python /path/to/inference-venv/bin/python \
  --inference-models sleepwalker/server/models
```

The second command checks technical compatibility, not predictive quality.
