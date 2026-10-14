from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from sleepwalker.server.services.annotation_service import AnnotationService
from sleepwalker.server.services.result_export_service import (
    ResultExportOptions,
    ResultExportService,
)
from sleepwalker.server.stores.annotation_store import AnnotationStore
from sleepwalker.server.stores.prediction_store import (
    PredictionKind,
    PredictionResult,
    PredictionStore,
)
from sleepwalker.server.stores.session_store import Session, SessionState
from sleepwalker.server.validation.study import SignalInfo, StudyFormat


def _session() -> Session:
    created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return Session(
        id="sess-test",
        created_at=created_at,
        expires_at=created_at + timedelta(hours=2),
        last_accessed_at=created_at,
        source_files=[Path("record.edf")],
        state=SessionState.READY,
        format=StudyFormat.EDF,
        duration=60.0,
        recording_start=datetime(2025, 1, 1),
        signals=[SignalInfo(id="EEG", label="EEG", unit="uV", sample_rate=200.0)],
    )


def test_builds_portable_result_payload_with_predictions_and_annotations() -> None:
    prediction_store = PredictionStore()
    prediction_store.add(
        PredictionResult(
            session_id="sess-test",
            model_id="sleep-model",
            task="sleep_stage",
            classes=("wake", "n1"),
            resolution=30.0,
            frame=pd.DataFrame(
                {
                    "time": [15.0],
                    "prob__wake": [0.8],
                    "prob__n1": [0.2],
                }
            ),
        )
    )
    annotation_service = AnnotationService(AnnotationStore())
    edit = annotation_service.set_label("sess-test", "sleep_stage", 0.0, "n1")
    service = ResultExportService(
        prediction_store,
        annotation_service,
        clock=lambda: datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc),
    )

    payload = service.build_payload(_session())

    assert payload["format"] == "sleepwalker-results"
    assert payload["version"] == 1
    assert payload["exported_at"] == "2025-01-02T12:00:00Z"
    assert payload["session"] == {
        "id": "sess-test",
        "created_at": "2025-01-01T00:00:00Z",
        "expires_at": "2025-01-01T02:00:00Z",
        "format": "edf",
        "duration": 60.0,
        "recording_start": "2025-01-01T00:00:00",
        "signals": [
            {"id": "EEG", "label": "EEG", "unit": "uV", "sample_rate": 200.0}
        ],
    }
    assert payload["tasks"] == [
        {
            "model_id": "sleep-model",
            "task": "sleep_stage",
            "kind": "epoch",
            "classes": ["wake", "n1"],
            "resolution": 30.0,
            "predictions": [
                {
                    "start": 0.0,
                    "end": 30.0,
                    "label": "wake",
                    "probabilities": {"wake": 0.8, "n1": 0.2},
                }
            ],
        }
    ]
    assert payload["annotations"] == [
        {
            "id": edit.id,
            "session_id": "sess-test",
            "operation": "set_label",
            "created_at": edit.created_at.isoformat().replace("+00:00", "Z"),
            "task": "sleep_stage",
            "start": 0.0,
            "label": "n1",
        }
    ]


def test_can_omit_probabilities_and_exports_continuous_hard_labels() -> None:
    prediction_store = PredictionStore()
    prediction_store.add(
        PredictionResult(
            session_id="sess-test",
            model_id="arousal-model",
            task="arousal",
            classes=("no_arousal", "arousal"),
            resolution=1.0,
            frame=pd.DataFrame(
                {
                    "time": [5.0],
                    "prob__no_arousal": [0.1],
                    "prob__arousal": [0.9],
                }
            ),
            kind=PredictionKind.CONTINUOUS,
        )
    )
    service = ResultExportService(prediction_store, AnnotationService(AnnotationStore()))

    payload = service.build_payload(
        _session(),
        ResultExportOptions(probabilities=False),
    )

    assert payload["tasks"][0]["predictions"] == [{"time": 5.0, "label": "arousal"}]


def test_can_exclude_optional_result_sections() -> None:
    service = ResultExportService(PredictionStore(), AnnotationService(AnnotationStore()))

    payload = service.build_payload(
        _session(),
        ResultExportOptions(predictions=False, annotations=False, metadata=False),
    )

    assert set(payload) == {"format", "version", "exported_at"}
