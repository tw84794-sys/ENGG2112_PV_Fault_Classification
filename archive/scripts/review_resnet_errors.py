"""Export ResNet18 validation errors for notebook review."""

import csv
import json
from collections import Counter
from pathlib import Path

from pv_fault.data import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    SolarModuleDataset,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs/resnet18-layer4-s2112-cpu-v1"


def write_csv(path, rows):
    columns = (
        "image_id",
        "group_id",
        "image_path",
        "true_label",
        "predicted_label",
        "p_normal",
        "p_predicted",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    dataset = SolarModuleDataset("val")
    predictions = json.loads(
        (RUN_DIR / "best_validation_predictions.json").read_text()
    )
    if len(predictions) != len(dataset):
        raise ValueError("Predictions do not cover the validation split.")

    normal_index = CLASS_TO_INDEX["No-Anomaly"]
    missed_faults = []
    false_alarms = []

    for saved, manifest_row in zip(predictions, dataset.rows):
        true_index = saved["true_index"]
        predicted_index = saved["predicted_index"]
        probabilities = saved["probabilities"]

        if (
            saved["image_id"] != manifest_row["image_id"]
            or saved["group_id"] != manifest_row["group_id"]
            or saved["image_path"] != manifest_row["image_path"]
            or true_index != CLASS_TO_INDEX[manifest_row["label"]]
            or len(probabilities) != len(CLASS_NAMES)
        ):
            raise ValueError("Prediction and validation manifest disagree.")

        review_row = {
            "image_id": saved["image_id"],
            "group_id": saved["group_id"],
            "image_path": saved["image_path"],
            "true_label": CLASS_NAMES[true_index],
            "predicted_label": CLASS_NAMES[predicted_index],
            "p_normal": probabilities[normal_index],
            "p_predicted": probabilities[predicted_index],
        }

        if true_index != normal_index and predicted_index == normal_index:
            missed_faults.append(review_row)
        elif true_index == normal_index and predicted_index != normal_index:
            false_alarms.append(review_row)

    missed_faults.sort(key=lambda row: row["p_normal"], reverse=True)
    false_alarms.sort(key=lambda row: row["p_predicted"], reverse=True)

    write_csv(RUN_DIR / "missed_faults_validation.csv", missed_faults)
    write_csv(RUN_DIR / "false_alarms_validation.csv", false_alarms)


    print("Faults predicted No-Anomaly:", len(missed_faults))
    print("Normal images predicted fault:", len(false_alarms))
    print("Missed faults by true class:")
    for label, count in Counter(
        row["true_label"] for row in missed_faults
    ).most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
