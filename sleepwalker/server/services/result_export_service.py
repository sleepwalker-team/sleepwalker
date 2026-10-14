from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from ..stores.annotation_store import AnnotationEdit
from ..stores.prediction_store import PredictionKind, PredictionResult, PredictionStore
from ..stores.session_store import Session
from .annotation_service import AnnotationService
from .prediction_mapper import (
    map_continuous_predictions,
    map_epoch_predictions,
    map_event_predictions,
)


@dataclass(frozen=True)
class ResultExportOptions:
    predictions: bool = True
    probabilities: bool = True
    annotations: bool = True
    metadata: bool = True


class ResultExportService:
    def __init__(
        self,
        prediction_store: PredictionStore,
        annotation_service: AnnotationService,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.prediction_store = prediction_store
        self.annotation_service = annotation_service
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def build_payload(
        self,
        session: Session,
        options: ResultExportOptions | None = None,
    ) -> dict:
        options = options or ResultExportOptions()
        payload = {
            "format": "sleepwalker-results",
            "version": 1,
            "exported_at": _format_datetime(self.clock()),
        }

        if options.metadata:
            payload["session"] = _session_metadata(session)

        if options.predictions:
            payload["tasks"] = [
                _prediction_task(result, session, options.probabilities)
                for result in self.prediction_store.list_for_session(session.id)
            ]

        if options.annotations:
            payload["annotations"] = [
                _annotation_edit(edit)
                for edit in self.annotation_service.list_edits(session.id)
            ]

        return payload


def _session_metadata(session: Session) -> dict:
    return {
        "id": session.id,
        "created_at": _format_datetime(session.created_at),
        "expires_at": _format_datetime(session.expires_at),
        "format": session.format.value if session.format is not None else None,
        "duration": session.duration,
        "recording_start": (
            _format_datetime(session.recording_start)
            if session.recording_start is not None
            else None
        ),
        "signals": [
            {
                "id": signal.id,
                "label": signal.label,
                "unit": signal.unit,
                "sample_rate": signal.sample_rate,
            }
            for signal in session.signals or []
        ],
    }


def _prediction_task(
    result: PredictionResult,
    session: Session,
    include_probabilities: bool,
) -> dict:
    if session.recording_start is None:
        raise ValueError("Cannot export predictions without a recording start time.")

    if result.kind == PredictionKind.EPOCH:
        predictions = map_epoch_predictions(result, session.recording_start)
        records = [
            _interval_prediction(prediction, include_probabilities)
            for prediction in predictions
        ]
    elif result.kind == PredictionKind.CONTINUOUS:
        predictions = map_continuous_predictions(result, session.recording_start)
        records = [
            _continuous_prediction(prediction, include_probabilities)
            for prediction in predictions
        ]
    else:
        predictions = map_event_predictions(result, session.recording_start)
        records = [
            _interval_prediction(prediction, include_probabilities)
            for prediction in predictions
        ]

    return {
        "model_id": result.model_id,
        "task": result.task,
        "kind": result.kind.value,
        "classes": list(result.classes),
        "resolution": result.resolution,
        "predictions": records,
    }


def _interval_prediction(prediction, include_probabilities: bool) -> dict:
    record = {
        "start": prediction.start,
        "end": prediction.end,
        "label": prediction.label,
    }

    if include_probabilities:
        record["probabilities"] = prediction.probabilities

    return record


def _continuous_prediction(prediction, include_probabilities: bool) -> dict:
    record = {"time": prediction.time}

    if include_probabilities:
        record["probabilities"] = prediction.probabilities
    else:
        record["label"] = max(
            prediction.probabilities,
            key=prediction.probabilities.__getitem__,
        )

    return record


def _annotation_edit(edit: AnnotationEdit) -> dict:
    values = {
        "id": edit.id,
        "session_id": edit.session_id,
        "operation": edit.operation.value,
        "created_at": _format_datetime(edit.created_at),
        "task": edit.task,
        "start": edit.start,
        "end": edit.end,
        "label": edit.label,
        "event_id": edit.event_id,
        "reverted_edit_id": edit.reverted_edit_id,
    }
    return {key: value for key, value in values.items() if value is not None}


def _format_datetime(value: datetime) -> str:
    text = value.isoformat()
    return f"{text[:-6]}Z" if text.endswith("+00:00") else text
