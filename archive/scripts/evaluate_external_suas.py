"""Exploratory normal-versus-fault check on an independent thermal archive.

This maps all non-No-Anomaly ResNet18 predictions to 'fault'. It does not
train or tune the model, and it is not a 12-class or unseen-site evaluation.
"""

import argparse
import hashlib
import io
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pv_fault.data import CLASS_NAMES
from pv_fault.resnet_spatial import configure_spatial_variant
from pv_fault.models.resnet import ResNet18Thermal, WEIGHT_SHA256


ARCHIVE_SHA256 = "2fa3ed0f5be5fe9200334a4f697c55fe5c15aa74b0f4f767f69634d785b73488"
FOLDERS = {
    "1": "Cell", "2": "Multi-Cell", "3": "Open-Circuit Substring",
    "4": "Open-Circuit", "5": "Open-Circuit Multi-Substring",
    "6": "Short-Circuit Substring", "7": "Short-Circuit", "8": "Normal",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/resnet18-layer4-s2112-cpu-v1/best_model.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/external-suas-resnet18-layer4-exploratory"))
    args = parser.parse_args()
    root = args.root.resolve()
    archive_path = root / "data/raw/external_suas_20860870.zip"
    weights_path = root / "data/pretrained/resnet18-f37072fd.pth"
    checkpoint_path = (args.checkpoint if args.checkpoint.is_absolute()
                       else root / args.checkpoint)
    output_dir = args.output if args.output.is_absolute() else root / args.output
    if output_dir.exists():
        raise FileExistsError(f"Result already exists: {output_dir}")
    if sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("External archive checksum changed.")
    if sha256(weights_path) != WEIGHT_SHA256:
        raise ValueError("Pretrained weight checksum changed.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model"] != "resnet18_layer4":
        raise ValueError("Expected a layer-4 ResNet18 checkpoint.")
    if tuple(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError("Checkpoint class order differs from this code.")
    model = ResNet18Thermal(len(CLASS_NAMES), weights_path, WEIGHT_SHA256)
    configure_spatial_variant(model, checkpoint.get("spatial_variant", "standard"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.set_num_threads(4)
    normal_index = CLASS_NAMES.index("No-Anomaly")

    records = []
    started = time.monotonic()
    with zipfile.ZipFile(archive_path) as archive, torch.inference_mode():
        names = sorted(name for name in archive.namelist()
                       if name.lower().endswith(".jpg"))
        if len(names) != 3675:
            raise ValueError("Unexpected external image count.")
        for start in range(0, len(names), 64):
            batch_names = names[start:start + 64]
            tensors = []
            folder_names = []
            for name in batch_names:
                parts = Path(name).parts
                if (len(parts) != 3 or parts[0] != "PV_Fault_Classification_Dataset"
                        or parts[1] not in FOLDERS):
                    raise ValueError(f"Unexpected ZIP member path: {name}")
                folder_names.append(parts[1])
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    if image.mode != "L":
                        raise ValueError(f"Expected grayscale image: {name}")
                    resized = image.resize((24, 40), Image.Resampling.BILINEAR)
                    pixels = np.asarray(resized, dtype=np.float32).copy() / 255.0
                    tensors.append(torch.from_numpy(pixels).unsqueeze(0))
            probabilities = model(torch.stack(tensors)).softmax(dim=1).cpu()
            for name, folder, probs in zip(batch_names, folder_names, probabilities):
                prediction = int(probs.argmax().item())
                records.append({
                    "image_path": name,
                    "external_label": FOLDERS[folder],
                    "true_binary": "normal" if folder == "8" else "fault",
                    "predicted_12_class": CLASS_NAMES[prediction],
                    "predicted_binary": "normal" if prediction == normal_index else "fault",
                    "no_anomaly_probability": float(probs[normal_index]),
                })

    normals = [row for row in records if row["true_binary"] == "normal"]
    faults = [row for row in records if row["true_binary"] == "fault"]
    summary = {
        "source": "https://zenodo.org/records/20860870",
        "archive_sha256": ARCHIVE_SHA256,
        "checkpoint_sha256": sha256(checkpoint_path),
        "preprocessing": "grayscale bilinear resize to 24x40, scale to [0,1]",
        "images": len(records),
        "normal": len(normals),
        "fault": len(faults),
        "normal_correct": sum(r["predicted_binary"] == "normal" for r in normals),
        "normal_false_alarms": sum(r["predicted_binary"] == "fault" for r in normals),
        "faults_detected": sum(r["predicted_binary"] == "fault" for r in faults),
        "faults_missed_as_normal": sum(r["predicted_binary"] == "normal" for r in faults),
        "normal_predictions_by_class": dict(Counter(
            r["predicted_12_class"] for r in normals
        )),
        "elapsed_seconds": time.monotonic() - started,
        "training_or_tuning_performed": False,
        "internal_test_evaluated": False,
        "limitation": (
            "Exploratory binary projection from 12 classes. One annotated "
            "orthomosaic, different fault taxonomy and preprocessing distribution; "
            "not independent unseen-site or production validation."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as target:
        for row in records:
            target.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    main()
