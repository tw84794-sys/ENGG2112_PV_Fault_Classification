"""Validation-only operating-point selection and fixed-policy test summaries."""
import math

import numpy as np
from pv_fault.data import CLASS_NAMES, CLASS_TO_INDEX
from pv_fault.evaluate import classification_metrics

NORMAL = CLASS_TO_INDEX["No-Anomaly"]


def screening_arrays(records):
    if not records:
        raise ValueError("Predictions must not be empty.")
    truth = np.asarray([row["true_index"] for row in records])
    probabilities = np.asarray([row["probabilities"] for row in records], dtype=float)
    if (truth.dtype.kind not in "iu" or np.any(truth < 0) or np.any(truth >= len(CLASS_NAMES))
            or probabilities.shape != (len(records), len(CLASS_NAMES))
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("Invalid class indices or probability vectors.")
    return truth != NORMAL, 1 - probabilities[:, NORMAL]


def screening_metrics(records, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError("Screening threshold must lie in [0, 1].")
    faults, scores = screening_arrays(records)
    flagged = scores >= threshold
    fault_count, normal_count = int(faults.sum()), int((~faults).sum())
    missed, alarms = int((faults & ~flagged).sum()), int((~faults & flagged).sum())
    return {"threshold": float(threshold), "fault_count": fault_count, "normal_count": normal_count,
            "missed_faults": missed, "normal_false_alarms": alarms,
            "recall": 1 - missed / fault_count if fault_count else None,
            "false_alarm_rate": alarms / normal_count if normal_count else None}


def select_threshold(validation_records, min_recall=.9, max_false_alarm_rate=.2):
    """Highest cutoff meeting recall minimises false alarms, including tied scores."""
    if not 0 < min_recall <= 1 or not 0 <= max_false_alarm_rate <= 1:
        raise ValueError("Invalid screening targets.")
    faults, scores = screening_arrays(validation_records)
    if not faults.any() or faults.all():
        raise ValueError("Threshold selection requires both normal and faulty validation images.")
    needed = math.ceil(min_recall * int(faults.sum()))
    threshold = np.sort(scores[faults])[::-1][needed - 1]
    metrics = screening_metrics(validation_records, threshold)
    return {"threshold": float(threshold), "validation": metrics,
            "targets_met": metrics["recall"] >= min_recall
                           and metrics["false_alarm_rate"] <= max_false_alarm_rate}


def group_intervals(records, threshold, *, resamples, seed, confidence):
    """Resample recorded related-image groups; this cannot establish site independence."""
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings.")
    truth = np.asarray([row["true_index"] for row in records])
    predicted = np.asarray([row["predicted_index"] for row in records])
    screening_arrays(records)
    classification_metrics(truth, predicted)
    if any(not row["group_id"] for row in records):
        raise ValueError("Every prediction needs a related-image group.")
    groups, inverse = np.unique([row["group_id"] for row in records], return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("accuracy", "macro_f1", "balanced_accuracy", "screening_recall", "screening_false_alarm_rate")}
    for _ in range(resamples):
        weights = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups))
        indices = np.repeat(np.arange(len(records)), weights[inverse])
        metrics = classification_metrics(truth[indices], predicted[indices])
        binary = screening_metrics([records[index] for index in indices], threshold)
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            samples[key].append(metrics[key])
        for key, value in (("screening_recall", binary["recall"]), ("screening_false_alarm_rate", binary["false_alarm_rate"])):
            if value is not None:
                samples[key].append(value)
    tail = (1 - confidence) / 2
    return {"method": "percentile bootstrap of recorded related-image groups", "groups": len(groups),
            "resamples": resamples, "seed": seed, "confidence": confidence,
            "bounds": {key: {"lower": float(np.quantile(values, tail)),
                              "upper": float(np.quantile(values, 1 - tail)),
                              "valid_resamples": len(values)} if values else None
                       for key, values in samples.items()}}
