# Optional Plotly/Dash transport

REST signal access remains unchanged and works without Plotly or Dash.
The optional `app.state.plotly_service` is a Python integration point, not
another `/api/v1` endpoint or an automatically launched UI.

Install the optional packages only when using this path:

```bash
pip install -r sleepwalker/server/requirements-plotly.txt
pytest -q tests/test_plotly_service.py tests/test_plotly_resampler.py
```

The real-resampler tests are skipped when the optional packages are absent;
the service/lifecycle tests still run without them.

## Two independent transports

- REST reads an EDF range in bounded chunks and returns reduced JSON points.
- Plotly loads the selected signals at original resolution into NumPy arrays
  and passes them as `hf_x`/`hf_y` to `FigureResampler`. Initial figures and
  zoom/pan patches contain resampled trace data, not the full arrays.

Both access the same session's source EDF and signal metadata. The Plotly
adapter deliberately does not use REST's already-downsampled points. It uses
the resampler's default selection algorithm, independently of REST's extrema
selection. A small output budget is not a guarantee that every spike survives.

## Python integration

The Dash callbacks must reference the same API application instance/session
service. Starting an unrelated Dash process does not share in-memory sessions.
Mounting/launching a concrete UI is a separate integration step.

```python
from dash import Dash, dcc, html

# api_app is the existing FastAPI application; session_id is an uploaded session.
plots = api_app.state.plotly_service
initial = plots.create_figure(session_id, ["EEG", "EOG"], view_id="signals", max_points=2000)

dash_app = Dash(__name__)
dash_app.layout = html.Div([
    dcc.Store(id="selected-session", data=session_id),
    dcc.Graph(id="signals", figure=initial),
])
plots.register_dash_callback(
    dash_app, graph_id="signals", session_store_id="selected-session", view_id="signals",
)
```

An application's selection callback creates/replaces the initial figure.
The registered zoom callback takes Graph `relayoutData` plus the selected
session ID, and returns `construct_update_data_patch(...)` output. It sends
`no_update` when the session/view no longer exists. Session switching must
replace the displayed initial figure; the browser does not store raw arrays.

`max_points` is per trace, between 3 and 100,000, and is a fixed shown-sample
budget rather than automatic measurement of the graph's pixel width. Times
are seconds relative to the recording start; each signal has its own subplot
and unit. The figure uses a separate lock for its mutable resampler state.

## Ownership and memory

Figures are keyed by `(session_id, view_id)`, not global variables. Recreating
the same view replaces it. Different sessions and views have independent
resamplers. Zoom/pan reads the retained arrays and does not reload the EDF.

Only selected signals are loaded. EDF reads are chunked, but the full NumPy
arrays stay in server RAM while the view exists: this transport optimization
is not a RAM-bounded whole-night reader. One float64 time/value pair takes
about 16 bytes per source sample before additional overhead. Equal-length,
equal-rate signals reuse the reader's time axis; separate views may load
separate arrays. No cache quota or eviction policy is imposed here.

Close a UI view explicitly to release its server-side state:

```python
plots.delete_figure(session_id, view_id="signals")
```

Session deletion (also with active inference jobs) immediately removes its
plot state. Expiration cleanup removes it too; zoom accesses renew session
TTL. Invalid/expired sessions reject subsequent updates. A deletion or view
replacement during a read/update cannot publish a stale figure/patch. Running
callbacks may retain local references until they finish. Application shutdown
clears all stored figures. REST behavior and inference parallelism are unchanged.
