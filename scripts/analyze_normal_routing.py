"""Summarize validation-only technician routing without changing class labels."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pv_fault.data import CLASS_NAMES


EXPECTED_PREDICTIONS_SHA256 = "ea764d15bf36296ac147e5e4c3f70795ec60677a3a4dacf61880c9d67297ac27"
EARLIER_PREDICTIONS_SHA256 = "bb9691cfafe26ca95766e9e3153980b65f3b9e68e9c86d65cdca3469f685d1b1"
THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    run_dir = root / "runs/resnet18-layer4-paperlr-s2112-cpu-v1"
    source = run_dir / "best_validation_predictions.json"
    if digest(source) != EXPECTED_PREDICTIONS_SHA256:
        raise ValueError("Validation predictions have changed.")
    earlier_source = (root / "runs/resnet18-layer4-s2112-cpu-v1"
                      / "best_validation_predictions.json")
    if digest(earlier_source) != EARLIER_PREDICTIONS_SHA256:
        raise ValueError("Earlier validation predictions have changed.")
    rows = json.loads(source.read_text())
    earlier_rows = json.loads(earlier_source.read_text())
    if len(rows) != 2988 or len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Expected 2,988 unique validation predictions.")
    if len(earlier_rows) != len(rows) or any(
        a["image_id"] != b["image_id"] or a["true_index"] != b["true_index"]
        for a, b in zip(rows, earlier_rows)
    ):
        raise ValueError("The two models' validation rows do not align.")
    normal_index = CLASS_NAMES.index("No-Anomaly")
    for row in rows:
        probabilities = row["probabilities"]
        if (len(probabilities) != len(CLASS_NAMES)
                or row["predicted_index"] != max(range(len(probabilities)),
                    key=lambda index: probabilities[index])
                or row["true_index"] not in range(len(CLASS_NAMES))):
            raise ValueError("Invalid saved prediction.")

    summary = {}
    for model_name, model_rows in (("earlier_lr", earlier_rows),
                                   ("cv_selected_lr", rows)):
        summary[model_name] = []
        for threshold in THRESHOLDS:
            fault_routed_normal = 0
            normal_sent_review = 0
            fault_by_class = Counter()
            for row in model_rows:
                is_normal = row["true_index"] == normal_index
                route_normal = (row["predicted_index"] == normal_index
                                and row["probabilities"][normal_index] >= threshold)
                if not is_normal and route_normal:
                    fault_routed_normal += 1
                    fault_by_class[CLASS_NAMES[row["true_index"]]] += 1
                if is_normal and not route_normal:
                    normal_sent_review += 1
            normal_accepted = 1496 - normal_sent_review
            normal_route_total = normal_accepted + fault_routed_normal
            summary[model_name].append({
                "normal_probability_threshold": threshold,
                "faults_routed_normal": fault_routed_normal,
                "faults_routed_normal_rate": fault_routed_normal / 1492,
                "normal_images_sent_to_review": normal_sent_review,
                "normal_review_rate": normal_sent_review / 1496,
                "total_images_sent_to_review": 2988 - normal_route_total,
                "fault_fraction_in_normal_route": (
                    fault_routed_normal / normal_route_total
                    if normal_route_total else None
                ),
                "faults_routed_normal_by_true_class": dict(sorted(fault_by_class.items())),
            })
    result = {
        "source_predictions_sha256": EXPECTED_PREDICTIONS_SHA256,
        "earlier_predictions_sha256": EARLIER_PREDICTIONS_SHA256,
        "source_split": "original validation only",
        "model_class_prediction_policy": "Keep original 12-class argmax and macro-F1 unchanged.",
        "routing_policy": (
            "Route as No-Anomaly candidate only when argmax is No-Anomaly "
            "and its softmax probability reaches threshold; otherwise route "
            "for qualified-technician review. A normal route is not a safety "
            "or reuse certificate."
        ),
        "thresholds": summary,
        "provisional_choice": 0.7,
        "choice_reason": (
            "Exploratory validation choice after comparing routing curves: "
            "at 0.7, the selected model routes 119/1492 faults normal and "
            "166/1496 normals to review; earlier model at argmax routed "
            "184/1492 faults normal and 162/1496 normals to review. "
            "Total selected-model review workload is 1539/2988 versus "
            "1470/2988. No expert risk or workload target has been supplied."
        ),
        "calibration_warning": "Softmax probabilities have not been calibrated.",
        "internal_test_evaluated": False,
    }
    output = args.output or run_dir / "validation-routing.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
