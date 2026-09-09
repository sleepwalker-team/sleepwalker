"""Convert classification outputs and optional labels into long-form prediction tables."""

import pandas as pd
import torch


def add_evaluation_columns(frame: pd.DataFrame, batch, selection, annotation_labels) -> pd.DataFrame:
    """Attach available ground truth, validity masks, and annotations to prediction rows."""
    if "target" in batch:
        frame["target"] = batch["target"][selection].argmax(dim=-1).reshape(-1).tolist()
    if "target_mask" in batch:
        frame["valid"] = batch["target_mask"][selection].reshape(-1).to(dtype=torch.bool).tolist()
    else:
        frame["valid"] = True
    if "annotation" in batch:
        if annotation_labels is None:
            raise ValueError("annotation_labels are required when a prediction batch contains annotations.")
        annotations = batch["annotation"][selection].reshape(-1, batch["annotation"].shape[-1])
        if annotations.shape[1] != len(annotation_labels):
            raise ValueError(f"Expected {len(annotation_labels)} annotation columns, got {annotations.shape[1]}.")
        for label_idx, label in enumerate(annotation_labels):
            frame[f"annotation__{label}"] = annotations[:, label_idx].tolist()
    return frame


def format_multiclass_prediction_batch(contract, batch, outputs, annotation_labels=None) -> pd.DataFrame:
    classes = list(contract["classes"])
    sequence_len = int(contract["sequence_len"])
    expected = (len(batch["patient"]), sequence_len, len(classes))
    if tuple(outputs.shape) != expected:
        raise ValueError(f"Expected logits shaped {expected}, got {tuple(outputs.shape)}.")

    probabilities = torch.softmax(outputs.detach().cpu(), dim=-1)
    patients = [str(patient) for patient in batch["patient"]]
    times = list(batch["time"])
    step_resolution = pd.to_timedelta(contract["target_resolution"]) / sequence_len
    target_offset = pd.to_timedelta(contract.get("target_offset", "0s"))
    prediction_indices = probabilities.argmax(dim=-1).reshape(-1).tolist()
    frame = pd.DataFrame({
        "patient": [patient for patient in patients for _ in range(sequence_len)],
        "task": str(contract["task"]),
        "time": [pd.Timestamp(time) + target_offset + step_idx * step_resolution for time in times for step_idx in range(sequence_len)],
        "prediction_idx": prediction_indices,
        "prediction": [classes[index] for index in prediction_indices],
    })
    flat_probabilities = probabilities.reshape(-1, len(classes))
    for class_idx, class_name in enumerate(classes):
        frame[f"prob__{class_name}"] = flat_probabilities[:, class_idx].tolist()
    return add_evaluation_columns(frame, batch, (...,), annotation_labels)


def format_multitask_prediction_batch(contract, batch, outputs, annotation_labels=None) -> pd.DataFrame:
    frames = []
    batch_size = len(batch["patient"])
    for task_idx, (task, config) in enumerate(contract["tasks"].items()):
        classes = list(config["classes"])
        n_steps = int(config["n_steps"])
        expected = (batch_size, n_steps, len(classes))
        if task not in outputs or tuple(outputs[task].shape) != expected:
            actual = None if task not in outputs else tuple(outputs[task].shape)
            raise ValueError(f"Expected logits for task '{task}' shaped {expected}, got {actual}.")
        probabilities = torch.softmax(outputs[task].detach().cpu(), dim=-1)
        resolution = pd.to_timedelta(config["target_resolution"])
        offset = pd.to_timedelta(config["target_offset"])
        prediction_indices = probabilities.argmax(dim=-1).reshape(-1).tolist()
        frame = pd.DataFrame({
            "patient": [str(patient) for patient in batch["patient"] for _ in range(n_steps)],
            "task": task,
            "time": [pd.Timestamp(time) + offset + step_idx * resolution for time in batch["time"] for step_idx in range(n_steps)],
            "prediction_idx": prediction_indices,
            "prediction": [classes[index] for index in prediction_indices],
        })
        flat_probabilities = probabilities.reshape(-1, len(classes))
        for class_idx, class_name in enumerate(classes):
            frame[f"prob__{class_name}"] = flat_probabilities[:, class_idx].tolist()
        selection = (slice(None), task_idx, slice(0, n_steps))
        frames.append(add_evaluation_columns(frame, batch, selection, annotation_labels))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def format_prediction_batch(contract, batch, outputs, annotation_labels=None) -> pd.DataFrame:
    contract_type = contract.get("type")
    if contract_type == "single-head-multiclass":
        if contract.get("task") is None:
            raise ValueError("A single-head classification contract must include task when formatting predictions.")
        return format_multiclass_prediction_batch(contract, batch, outputs, annotation_labels=annotation_labels)
    if contract_type == "multitask":
        return format_multitask_prediction_batch(contract, batch, outputs, annotation_labels=annotation_labels)
    raise ValueError(f"Unsupported classification contract type {contract_type!r}.")
