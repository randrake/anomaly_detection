# evaluation.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def _print_metrics_table(metrics, title="Evaluation Metrics"):
    """
    Pretty printer for evaluation metrics.
    Matches the aligned style used for true → pred inspection.
    """
    print(f"\n{title}")
    print("───────────────")
    print(" metric      │ value")
    print("─────────────┼────────")

    order = ["accuracy", "precision", "recall", "f1", "macro_f1"]
    for k in order:
        v = metrics[k]
        print(f" {k:<11} │ {v:.4f}")


def evaluate_model(
    y_true,
    y_pred,
    *,
    task_type="multiclass",
    average="macro",
    return_report=False,
    verbose=True
):
    """
    Unified evaluation function for anomaly detection and classification.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ---------- metrics ----------
    if task_type == "binary":
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="binary", zero_division=0)
        macro_f1  = f1
    else:
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall    = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1        = f1_score(y_true, y_pred, average=average, zero_division=0)
        macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "accuracy": {accuracy: .2%f},
        "precision": {precision: .2%f},
        "recall": {recall: .2%f},
        "f1": {f1: .2%f},
        "macro_f1": {macro_f1: .2%f}
    }

    # ---------- pretty print ----------
    if verbose:
        _print_metrics_table(metrics)

    if return_report:
        report = classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
        return metrics, report

    return metrics
