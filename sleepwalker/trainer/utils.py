import json
import os
import random
import tempfile
from typing import List, Optional
import numpy as np
import pandas as pd
import torch

from sleepwalker.models.Basemodel import BaseModel 


def trim_wake(
    data_df: pd.DataFrame,
    label_df: Optional[pd.DataFrame],
    label_extra_df: Optional[pd.DataFrame],
    wake_label: str = "wake",
) -> Optional[tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    if label_df is None or len(label_df) == 0:
        return None

    label_df = label_df.sort_values(["Starttime", "Endtime"]).reset_index(drop=True)
    non_wake = label_df["Label"] != wake_label
    if not non_wake.any():
        return None

    non_wake_positions = np.flatnonzero(non_wake.to_numpy())
    first_idx = int(non_wake_positions[0])
    last_idx = int(non_wake_positions[-1])
    label_df = label_df.iloc[first_idx:last_idx + 1].copy()

    lower = label_df["Starttime"].min()
    upper = label_df["Endtime"].max()

    if label_extra_df is not None:
        label_extra_df = label_extra_df.copy()
        label_extra_df = label_extra_df[
            (label_extra_df["Endtime"] > lower) & (label_extra_df["Starttime"] < upper)
        ]
        if len(label_extra_df) > 0:
            label_extra_df["Starttime"] = label_extra_df["Starttime"].clip(lower=lower, upper=upper)
            label_extra_df["Endtime"] = label_extra_df["Endtime"].clip(lower=lower, upper=upper)
            label_extra_df = label_extra_df[label_extra_df["Endtime"] > label_extra_df["Starttime"]]
        else:
            label_extra_df = None

    return label_df, label_extra_df

def read_jsonl(filename: str) -> pd.DataFrame:
    """
    Reads a JSON Lines (.jsonl) file and returns its contents as a pandas DataFrame.

    Each line in a .jsonl file is expected to be a valid JSON object.
    This function parses each line into a dictionary and constructs a DataFrame
    from the collection of dictionaries.

    Parameters
    ----------
    filename : str
        Path to the JSON Lines file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the parsed contents of the file, with one row per line.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If any line in the file is not valid JSON.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(data)
    

def append_to_jsonl(filename: str, record: dict):
    """
    Append a dictionary record to a JSONL file.
    This function serializes a Python dictionary to a JSON string and appends it as a new line
    to a JSONL file. It handles NumPy arrays by converting them to lists.
    Parameters
    ----------
    filename : str
        The base filename without the .jsonl extension. The function will append '.jsonl'
        to this name.
    record : dict
        The dictionary to append to the file. Can contain NumPy arrays which will be 
        automatically converted to lists.
    Notes
    -----
    - If the file doesn't exist, it will be created.
    - Uses a custom JSONEncoder to handle NumPy arrays.
    - Each record is written as a separate line in the file.
    """

    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.bool_):
                return bool(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, pd.Timedelta):
                return str(o)
            elif isinstance(o, pd.Timestamp):
                return o.isoformat()
            else:
                return super(NumpyEncoder, self).default(o)

    with open(f"{filename}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + "\n")

def store_checkpoint(model: BaseModel, optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], folder:str = tempfile.mkdtemp(prefix="sleepwalker_")) -> str:
    """
    Saves model, optimizer, and scheduler (if provided) states to the specified folder.
    Args:
        model (BaseModel): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler to save.
            If None, no scheduler state will be saved.
        folder (str, optional): Directory path where checkpoint files will be saved.
            Defaults to system's temporary directory.
    Returns:
        str: The folder path where the checkpoint files were saved.
    Note:
        The function saves the model state as "model.pt", optimizer state as "optimizer.pt",
        and scheduler state (if provided) as "scheduler.pt" in the specified folder.
    """

    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(folder, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(folder, "optimizer.pt"))
    if scheduler:
        torch.save(scheduler.state_dict(), os.path.join(folder, "scheduler.pt"))

    return folder

def cohen_kappa_from_confusion_matrix(confusion_matrix):
    """
    Calculate Cohen's Kappa coefficient from a confusion matrix.
    Cohen's Kappa is a statistical measure of inter-rater agreement or reliability.
    It accounts for the possibility of agreement occurring by chance.
    Args:
        confusion_matrix (numpy.ndarray): A square confusion matrix where the rows 
                                           represent the true classes and the columns 
                                           represent the predicted classes.
    Returns:
        float: The Cohen's Kappa coefficient. A value between -1 and 1, where:
               - 1 indicates perfect agreement,
               - 0 indicates agreement equivalent to chance,
               - Negative values indicate less agreement than expected by chance.
    """

    n_total = np.sum(confusion_matrix)
    
    # Observed agreement (P_o)
    p_o = np.trace(confusion_matrix) / n_total
    
    # Expected agreement (P_e)
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    p_e = np.sum(row_sums * col_sums) / (n_total ** 2)
    
    if p_o < p_e or p_e > 1 or np.abs(1 - p_e) < 1e-3:
        kappa = 0.0
    else: 
        kappa = (p_o - p_e) / (1 - p_e)
    
    return kappa

def f1_score_from_confusion_matrix(confusion_matrix, macro=False) -> float:
    """
    Calculate the F1-score (macro or micro) from a given confusion matrix.
    Parameters:
    -----------
    confusion_matrix : numpy.ndarray
        A square matrix of shape (num_classes, num_classes) representing the confusion matrix.
        Rows correspond to the true classes, and columns correspond to the predicted classes.
    macro : bool, optional
        If True, computes the macro F1-score (averaged across all classes).
        If False, computes the micro F1-score (global F1-score across all samples).
        Default is False.
    Returns:
    --------
    float
        The computed F1-score (macro or micro, depending on the `macro` parameter).
    Notes:
    ------
    - The macro F1-score is the unweighted mean of F1-scores for each class.
    - The micro F1-score aggregates contributions of all classes to compute the F1-score globally.
    - Handles division by zero by assigning a score of 0.0 when precision or recall cannot be computed.
    """
    
    num_classes = confusion_matrix.shape[0]
    
    # Compute macro F1-score:
    if macro:
        # Compute precision, recall, and F1-score for each class
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1_score = np.zeros(num_classes)
        
        for i in range(num_classes):
            true_positives = confusion_matrix[i, i]
            false_positives = np.sum(confusion_matrix[:, i]) - true_positives
            false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
            
            # Check for division by zero
            if true_positives == 0 and false_positives == 0:
                precision[i] = 0.0
            else:
                precision[i] = true_positives / (true_positives + false_positives)
            
            if true_positives == 0 and false_negatives == 0:
                recall[i] = 0.0
            else:
                recall[i] = true_positives / (true_positives + false_negatives)
            
            # Calculate F1-score
            if precision[i] == 0.0 and recall[i] == 0.0:
                f1_score[i] = 0.0
            else:
                f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        
        macro_f1 = np.mean(f1_score)
        return macro_f1
    
    # Compute micro F1-score
    else:
        total_true_positives = np.sum(np.diagonal(confusion_matrix))
        total_false_positives = np.sum(confusion_matrix, axis=0) - np.diagonal(confusion_matrix)
        total_false_negatives = np.sum(confusion_matrix, axis=1) - np.diagonal(confusion_matrix)
        
        # Check for division by zero
        if total_true_positives == 0 and np.sum(total_false_positives) == 0:
            micro_precision = 0.0
        else:
            micro_precision = total_true_positives / (total_true_positives + np.sum(total_false_positives))
        
        if total_true_positives == 0 and np.sum(total_false_negatives) == 0:
            micro_recall = 0.0
        else:
            micro_recall = total_true_positives / (total_true_positives + np.sum(total_false_negatives))
        
        # Calculate micro F1-score
        if micro_precision == 0.0 and micro_recall == 0.0:
            micro_f1 = 0.0
        else:
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        
        return micro_f1

def target_to_multiclass(target, default_idx, min_event_seconds, raise_error=True):
    """
    Simplified: target is a 1D tensor of size (num_classes,).
    Returns a one-hot tensor based on threshold.
    """
    num_classes = target.shape[0]

    active = target > min_event_seconds
    active_sum = active.sum().item()

    if raise_error and active_sum > 1:
        raise ValueError("Multiple active classes found.")

    out = torch.zeros(num_classes, dtype=torch.float)

    if active_sum == 1:
        idx = active.nonzero(as_tuple=False).item()
        out[idx] = 1.0
    elif active_sum == 0:
        if default_idx is not None:
            out[default_idx] = 1.0
        elif raise_error:
            raise ValueError("Ambiguous class labels found with no active class and no default_idx.")

    return out

def get_target_as_multiclass(target, target_extra = None, percentage:float = 0.5, class_cnts:Optional[List[float]] = None):
    try:
        freq = pd.to_timedelta(target.index.freq).total_seconds()
        targets = torch.tensor(target.sum().to_numpy())
        target = target_to_multiclass(targets, None, len(target)*freq*percentage, True)

        item = {"target":target}

        if class_cnts and len(class_cnts) == len(target):
            probas = class_cnts / np.sum(class_cnts)
            m = min(probas)
            if random.random() > m / probas[target.argmax()]:
                return None

        if target_extra is not None:
            freq = pd.to_timedelta(target_extra.index.freq).total_seconds()
            targets = torch.tensor(target_extra.sum().to_numpy())
            target_extra = target_to_multiclass(targets, None, len(target_extra)*freq*percentage, True)
            item["target_extra"] = target_extra

        return item
    except Exception as e:
        pass
    return None
