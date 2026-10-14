# Workspace archive, version 1

Request `POST /api/v1/sessions/{session_id}/exports` with
`{"format": "sleepwalker", "mode": "workspace"}`. The export runs as a job.
Download the ready export as `<export_id>.sleepwalker.zip`.

The ZIP contains:

- `manifest.json`: `format: "sleepwalker"`, `format_version: 1`, creation time,
  producing Sleepwalker version, source filenames, task IDs, prediction-file
  mapping, model metadata and SHA-256 checksums.
- `study/metadata.json`: the original session ID and timestamps, study format,
  recording start, duration and signal metadata. Server filesystem paths are
  excluded. An import assigns a new session ID and new expiry time.
- `study/source/<filename>`: the original uploaded study files, copied as bytes.
- `predictions/0000.json`, etc.: one task per document, identified through the
  manifest's `prediction_files` mapping. Each document contains model ID, task,
  kind, classes, resolution and normalized predictions with all probabilities.
  Times are seconds relative to `recording_start`. Epoch/event records contain
  `start`, `end`, `label` and `probabilities`; continuous records contain `time`
  and `probabilities`. These are the API's normalized results, not raw internal
  DataFrame layouts.
- `annotations/edits.json`: all edits in insertion order, including deletions
  and reverts. Predictions and human edits remain separate.
- `provenance/models.json`: model IDs and available descriptive metadata.
  Model packages and their local filesystem paths are excluded. The current
  registry has no model-version field, so none is invented in the archive.

The manifest's `checksums` object maps every archive member except
`manifest.json` to the SHA-256 digest of its uncompressed bytes. This detects
corruption; it is not an authenticity signature. All structured documents are
UTF-8 JSON. A workspace always includes all sections and probabilities;
partial exports are available through `json/results` instead.

The exporter streams source files in chunks and checks job cancellation
between chunks. It publishes the finished archive only after closing it.
Cancellation or failure removes partial files, and session cleanup removes
both the archive and its export record.

## Import

Upload the ZIP as the multipart field `bundle` to
`POST /api/v1/sessions/import`. The response is `202` with `session_id`,
`job_id` and `state: "processing"`. Poll `/api/v1/jobs/{job_id}` for completion
and `/api/v1/sessions/{session_id}` for the restored session.

The initial upload checks archive paths, member types, size limits, required
members, manifest shape and format version. Invalid archives return
`400 INVALID_BUNDLE`, oversized uploads `413 FILE_TOO_LARGE`, and unsupported
versions `422 UNSUPPORTED_FORMAT_VERSION`. The default uncompressed total
limit equals the upload limit (10 GiB); the manifest limit is 1 MiB, each other
JSON document is limited to 256 MiB, and at most 10,000 members are allowed.

The import job checks every member's SHA-256 checksum, validates JSON contents,
streams source files into a new session directory and revalidates the study
using the existing EDF/BDF reader. Source metadata must match the archive.
Prediction documents are converted back to internal result tables. Edit and
event IDs are regenerated, and all references between edits are remapped.
Model provenance is retained for later re-export. Importing results does not
require installed model packages and does not register models for inference.

On success the session becomes `ready` with a fresh inactivity deadline.
Checksum/content failures appear as `INVALID_BUNDLE` in the failed job;
restored files and results are removed. A failed or cancelled import leaves a
`failed` session and its job available for inspection until deletion or expiry.
Deleting a session during import requests cancellation and cleans up after
the worker stops. Archive contents are never executed or unpickled.
