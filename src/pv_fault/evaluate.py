"""Class-aware metrics for solar-module fault predictions."""

import numpy as np  # noqa: E402

from pv_fault.data import CLASS_NAMES  # noqa: E402


def classification_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    class_names = tuple(class_names)
    if "No-Anomaly" not in class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique and include No-Anomaly.")

    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if true.ndim != 1 or predicted.ndim != 1 or len(true) != len(predicted):
        raise ValueError("Labels must be equal-length, one-dimensional arrays.")
    if len(true) == 0:
        raise ValueError("Cannot score an empty set of predictions.")
    if true.dtype.kind not in "iu" or predicted.dtype.kind not in "iu":
        raise ValueError("Class indices must be integers.")

    class_count = len(class_names)
    if (
        np.any(true < 0) or np.any(true >= class_count)
        or np.any(predicted < 0) or np.any(predicted >= class_count)
    ):
        raise ValueError("Class index is outside the class-name list.")

    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (true, predicted), 1)

    correct = np.diag(confusion)
    actual_count = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)

    precision = np.divide(
        correct, predicted_count,
        out=np.zeros(class_count, dtype=float),
        where=predicted_count > 0,
    )
    recall = np.divide(
        correct, actual_count,
        out=np.zeros(class_count, dtype=float),
        where=actual_count > 0,
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(class_count, dtype=float),
        where=(precision + recall) > 0,
    )

    normal_index = class_names.index("No-Anomaly")
    fault_classes = np.arange(class_count) != normal_index
    fault_count = int(actual_count[fault_classes].sum())
    normal_count = int(actual_count[normal_index])
    faults_called_normal = int(confusion[fault_classes, normal_index].sum())
    normal_called_fault = int(confusion[normal_index, fault_classes].sum())

    return {
        "count": len(true),
        "accuracy": float(correct.sum() / len(true)),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            name: {
                "support": int(actual_count[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, name in enumerate(class_names)
        },
        "faults_called_normal": faults_called_normal,
        "fault_to_normal_rate": (
            faults_called_normal / fault_count if fault_count else None
        ),
        "normal_called_fault": normal_called_fault,
        "normal_false_alarm_rate": (
            normal_called_fault / normal_count if normal_count else None
        ),
    }