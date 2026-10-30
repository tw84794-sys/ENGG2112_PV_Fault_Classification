"""Grayscale datasets and batches using the frozen, duplicate-aware split."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_MANIFEST = ROOT / "data/splits/infrared_solar_modules_v1.csv"
CLASS_NAMES = (
    "Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi", "Hot-Spot",
    "Hot-Spot-Multi", "No-Anomaly", "Offline-Module", "Shadowing", "Soiling",
    "Vegetation",
)
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
SPLITS = ("train", "val", "test")


def _verified_rows(manifest, data_root):
    summary = json.loads(manifest.with_suffix(".json").read_text())
    metadata_path = data_root / "processed/near_clean/module_metadata.json"
    for path, key in ((manifest, "manifest_sha256"),
                      (metadata_path, "metadata_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary[key]:
            raise ValueError(f"Fingerprint mismatch for {path}; verify the dataset.")
    metadata = json.loads(metadata_path.read_text())
    with manifest.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["image_id", "image_path", "label", "group_id", "split"]:
            raise ValueError("Unexpected split manifest columns.")
        rows = list(reader)
    seen = set()
    group_splits = {}
    labels_by_split = {split: set() for split in SPLITS}
    for row in rows:
        image_id = row["image_id"]
        if image_id in seen or image_id not in metadata:
            raise ValueError(f"Repeated or unknown image ID: {image_id}")
        seen.add(image_id)
        relative = Path(row["image_path"])
        if (relative.is_absolute() or len(relative.parts) != 2
                or relative.parts[0] != "images" or ".." in relative.parts):
            raise ValueError(f"Unsafe image path: {relative}")
        path = data_root / "raw" / relative
        if not path.resolve().is_relative_to((data_root / "raw").resolve()):
            raise ValueError(f"Image outside dataset: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        details = metadata[image_id]
        if (row["label"] not in CLASS_TO_INDEX
                or row["label"] != details["anomaly_class"]
                or row["image_path"] != details["image_filepath"]):
            raise ValueError(f"Image path or label mismatch for {image_id}")
        split, group = row["split"], row["group_id"]
        if split not in SPLITS or not group:
            raise ValueError(f"Invalid split or group for {image_id}")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Related-image group {group} crosses splits.")
        group_splits[group] = split
        labels_by_split[split].add(row["label"])
    if seen != set(metadata) or len(rows) != summary["images"]:
        raise ValueError("Manifest does not cover the expected dataset.")
    if any(labels != set(CLASS_NAMES) for labels in labels_by_split.values()):
        raise ValueError("Every split must contain all expected classes.")
    return rows


class SolarModuleDataset(Dataset):
    """Return (float32 image [1, 40, 24], integer class index).

    Pixels are scaled from [0, 255] to [0, 1], without resizing or RGB
    conversion. Optional augmentation receives a tensor and is train-only.
    Run create_split.py --verify-only before experiments to verify image bytes.
    """

    class_names = CLASS_NAMES
    class_to_index = CLASS_TO_INDEX

    def __init__(self, split, *, data_root=DEFAULT_DATA_ROOT,
                 manifest=DEFAULT_MANIFEST, transform=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if transform is not None and split != "train":
            raise ValueError("Augmentation is only allowed on the training split.")
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.rows = [row for row in _verified_rows(Path(manifest), self.data_root)
                     if row["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_root / "raw" / row["image_path"]
        with Image.open(path) as image:
            if image.mode != "L" or image.size != (24, 40):
                raise ValueError(f"Expected grayscale 24 x 40 image: {path}")
            pixels = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).unsqueeze(0)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, CLASS_TO_INDEX[row["label"]]


def _seed_worker(worker_id):
    # Seed Python/NumPy augmentations from PyTorch's per-worker seed.
    import random
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_data_loaders(*, batch_size=64, seed=2112, num_workers=0,
                      data_root=DEFAULT_DATA_ROOT, manifest=DEFAULT_MANIFEST,
                      train_transform=None):
    """Build loaders; shuffle training only and retain every final partial batch.

    The seed controls sampling and worker RNGs. With num_workers=0, callers
    should seed global RNGs themselves if supplying random transforms.
    """
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    loaders = {}
    for split in SPLITS:
        dataset = SolarModuleDataset(
            split, data_root=data_root, manifest=manifest,
            transform=train_transform if split == "train" else None,
        )
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=split == "train",
            num_workers=num_workers, drop_last=False,
            generator=torch.Generator().manual_seed(seed),
            worker_init_fn=_seed_worker,
        )
    return loaders
