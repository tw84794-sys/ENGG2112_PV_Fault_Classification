"""Export ResNet18 validation errors and create an image review gallery."""

import base64
import csv
import html
import json
from collections import Counter
from pathlib import Path

from pv_fault.data import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    SolarModuleDataset,
)


ROOT = Path(__file__).resolve().parents[1]
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


def make_image_card(row, data_root):
    image_path = (data_root / row["image_path"]).resolve()

    if not image_path.is_relative_to(data_root.resolve()):
        raise ValueError(f"Image path leaves the dataset folder: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")

    image_id = html.escape(str(row["image_id"]))
    true_label = html.escape(row["true_label"])
    predicted_label = html.escape(row["predicted_label"])
    p_normal = float(row["p_normal"])

    return f"""
    <article class="card">
      <img src="data:image/jpeg;base64,{image_data}" alt="Image {image_id}">
      <div><strong>Image {image_id}</strong></div>
      <div>Dataset label: {true_label}</div>
      <div>Model prediction: {predicted_label}</div>
      <div>No-Anomaly probability: {p_normal:.1%}</div>
    </article>
    """


def write_gallery(path, data_root, missed_faults, false_alarms):
    sections = []

    for title, rows in (
        ("Faults predicted No-Anomaly", missed_faults),
        ("Normal images predicted as faults", false_alarms),
    ):
        cards = "\n".join(
            make_image_card(row, data_root) for row in rows
        )
        sections.append(
            f"<h2>{html.escape(title)} ({len(rows)})</h2>"
            f'<div class="grid">{cards}</div>'
        )

    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ResNet18 validation error review</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 14px;
    }
    .card {
      border: 1px solid #bbb;
      border-radius: 8px;
      padding: 12px;
      line-height: 1.5;
    }
    img {
      display: block;
      width: 192px;
      height: 320px;
      object-fit: contain;
      image-rendering: pixelated;
      background: #eee;
      margin-bottom: 8px;
    }
  </style>
</head>
<body>
  <h1>ResNet18 validation error review</h1>
  <p>These are enlarged 24 × 40 thermal images. Probabilities are not
  calibrated risk scores. Record uncertain labels for discussion; do not
  change or delete an image solely because the model disagrees.</p>
"""
    page += "\n".join(sections)
    page += "\n</body>\n</html>\n"

    path.write_text(page, encoding="utf-8")


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

    gallery_path = RUN_DIR / "validation_error_gallery.html"
    write_gallery(
        gallery_path,
        dataset.data_root,
        missed_faults,
        false_alarms,
    )

    print("Faults predicted No-Anomaly:", len(missed_faults))
    print("Normal images predicted fault:", len(false_alarms))
    print("Missed faults by true class:")
    for label, count in Counter(
        row["true_label"] for row in missed_faults
    ).most_common():
        print(f"  {label}: {count}")
    print("Saved gallery to", gallery_path)


if __name__ == "__main__":
    main()