# evaluation.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def evaluate_model(
    y_true,
    y_pred,
    *,
    task_type="multiclass",
    average="macro",
    return_report=False
):
    """
    Unified evaluation function for anomaly detection and classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    task_type : str
        "binary" or "multiclass".
    average : str
        Averaging method for precision/recall/F1.
        Use:
            - "binary" for binary anomaly detection
            - "macro" for imbalanced multiclass (recommended for 3W)
    return_report : bool
        If True, returns sklearn classification_report string.

    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, precision, recall, f1, macro_f1.
    report : str (optional)
        Full classification report.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task_type == "binary":
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        macro_f1 = f1  # same for binary
    else:
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1
    }

    if return_report:
        report = classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
        return metrics, report

    return metrics
