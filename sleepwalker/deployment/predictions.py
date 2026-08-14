"""Convert classification outputs into stable prediction tables."""

import pandas as pd
import torch


def format_multiclass_prediction_batch(contract, batch, outputs, target_resolution) -> pd.DataFrame:
    classes = list(contract["classes"])
    sequence_len = int(contract["sequence_len"])
    expected = (sequence_len, len(classes))
    if outputs.ndim != 3 or tuple(outputs.shape[1:]) != expected:
        raise ValueError(f"Expected logits shaped (B, {expected[0]}, {expected[1]}), got {tuple(outputs.shape)}.")

    probabilities = torch.softmax(outputs.detach().cpu(), dim=-1)
    prediction_indices = probabilities.argmax(dim=-1)
    patients = list(batch["patient"])
    times = list(batch["time"])
    if len(patients) != probabilities.shape[0] or len(times) != probabilities.shape[0]:
        raise ValueError(f"Expected {probabilities.shape[0]} patient paths and timestamps, got {len(patients)} and {len(times)}.")

    step_resolution = pd.to_timedelta(target_resolution) / sequence_len
    frame = {
        "patient": [patient for patient in patients for _ in range(sequence_len)],
        "time": [pd.Timestamp(time) + step_idx * step_resolution for time in times for step_idx in range(sequence_len)],
        "prediction_idx": prediction_indices.reshape(-1).tolist(),
    }
    frame["prediction"] = [classes[idx] for idx in frame["prediction_idx"]]
    flat_probabilities = probabilities.reshape(-1, len(classes))
    for class_idx, class_name in enumerate(classes):
        frame[f"prob__{class_name}"] = flat_probabilities[:, class_idx].tolist()
    return pd.DataFrame(frame)


def format_multitask_prediction_batch(contract, batch, outputs) -> pd.DataFrame:
    rows = []
    for sample_idx, target_start in enumerate(batch["time"]):
        row = {"patient": batch["patient"][sample_idx], "time": target_start}
        target_start = pd.Timestamp(target_start)
        for task, config in contract["tasks"].items():
            probabilities = torch.softmax(outputs[task][sample_idx].detach().cpu(), dim=-1)
            prediction_indices = probabilities.argmax(dim=-1)
            task_resolution = pd.to_timedelta(config["target_resolution"])
            task_start = target_start + pd.to_timedelta(config["target_offset"])
            for step_idx in range(int(config["n_steps"])):
                suffix = "" if int(config["n_steps"]) == 1 else f"__step_{step_idx}"
                prediction_idx = int(prediction_indices[step_idx].item())
                row[f"{task}{suffix}__time"] = task_start + step_idx * task_resolution
                row[f"{task}{suffix}__prediction_idx"] = prediction_idx
                row[f"{task}{suffix}__prediction"] = config["classes"][prediction_idx]
                for label_idx, label in enumerate(config["classes"]):
                    row[f"{task}{suffix}__prob__{label}"] = float(probabilities[step_idx, label_idx].item())
        rows.append(row)
    return pd.DataFrame(rows)


def format_prediction_batch(contract, batch, outputs, target_resolution) -> pd.DataFrame:
    contract_type = contract.get("type")
    if contract_type == "single-head-multiclass":
        return format_multiclass_prediction_batch(contract, batch, outputs, target_resolution)
    if contract_type == "multitask":
        return format_multitask_prediction_batch(contract, batch, outputs)
    raise ValueError(f"Unsupported classification contract type {contract_type!r}.")
