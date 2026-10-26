# Reading human annotations

`GET /api/v1/sessions/{session_id}/annotations` returns:

```json
{
  "edits": [],
  "labels": [],
  "events": []
}
```

All entries use the same fields as the existing POST response, including
`edit_id`, `operation`, `created_at`, `task`, `start`, `end`, `label`,
`event_id` and `reverted_edit_id`. Inapplicable fields are `null`.

- `edits` is the complete history in insertion order, including overwritten
  labels, deleted events and revert operations.
- `labels` contains the latest non-reverted `set_label` edit for each
  `(task, start)`. Reverting the latest edit restores an earlier active edit;
  reverting all edits leaves no human override for that epoch.
- `events` contains active human-added events. Deleted events and reverted
  additions are excluded. Reverting a deletion restores the event, unless
  its addition was also reverted.

The three lists come from a single history snapshot. An unannotated ready
session returns three empty lists. Unknown, deleted or expired sessions return
`404 SESSION_NOT_FOUND`; sessions that are not ready return
`409 INVALID_SESSION_STATE`.

The predictions endpoint continues to return the original model results.
Clients can apply `labels` as overrides and display `events` as an annotation
track, without needing to replay the edit history themselves. Human-added
events are separate from model-generated event predictions.

The endpoint also works after workspace import. Imported edits and events
have new IDs, with references remapped by the importer, but their effective
state and history are retained.

```bash
curl http://127.0.0.1:6006/api/v1/sessions/SESSION_ID/annotations
```

Replace `SESSION_ID` with the actual ID returned by session creation or import.
