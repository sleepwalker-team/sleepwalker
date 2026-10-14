# Job updates with Server-Sent Events

Connect to `GET /api/v1/jobs/{job_id}/events`:

```bash
curl -N http://127.0.0.1:6006/api/v1/jobs/JOB_ID/events
```

The response is `text/event-stream`. It immediately sends the current job
snapshot, then complete updated snapshots using the same public fields as
`GET /api/v1/jobs/{job_id}`:

```text
event: job_update
data: {"id":"job_example","type":"inference","state":"running","phase":"inference","progress":0.42,"message":"Running inference","updated_at":"2026-10-14T12:00:00Z","error":null}

```

`progress: null` means no numerical progress is available. Failed jobs include
the same structured `error` as the polling endpoint. The stream sends the
terminal snapshot and closes on `succeeded`, `failed` or `cancelled`. Connecting
to an already terminal job sends exactly one snapshot and closes.

An unknown job returns the normal `404 JOB_NOT_FOUND` JSON response before
streaming starts. If a job is deleted while the stream is open, the server
sends `event: job_deleted` with the normal error envelope and closes; an HTTP
status cannot be changed after streaming has started. Normal cancellation
sends `cancelling` and then `cancelled` through `job_update`.

Idle connections receive an SSE comment, `: keep-alive`, every 15 seconds.
Disconnecting a client removes its subscription but does not cancel the job.
Several clients may independently observe the same job.

Subscriptions hold the initial snapshot and at most 64 pending updates. Slow
clients may miss intermediate progress snapshots, but the latest state and
the terminal snapshot are preserved. Reconnecting starts with the current
state; this V1 does not offer an event history or `Last-Event-ID` replay.

Browser clients should listen for the named `job_update` event. Close their
`EventSource` on a terminal state or `job_deleted`, since otherwise its normal
automatic reconnect behavior would reopen a completed stream. Polling remains
available for clients that do not use SSE. State and subscriptions remain
local to the single server process, as with the existing in-memory stores.
