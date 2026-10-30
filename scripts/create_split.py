"""Create and verify a frozen, group-aware train/validation/test manifest."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "near_clean"
IMAGE_ROOT = ROOT / "data/raw"
METADATA_PATH = DATA_ROOT / "module_metadata.json"
GROUPS_PATH = DATA_ROOT / "related_groups.csv"
DEFAULT_OUTPUT = ROOT / "data" / "splits" / "infrared_solar_modules_v1.csv"
FIELDS = ("image_id", "image_path", "label", "group_id", "split")
SPLITS = ("train", "val", "test")
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Near-clean metadata must be a nonempty object.")

    with GROUPS_PATH.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["image_id", "label", "group_id", "group_size"]:
            raise ValueError("Unexpected columns in related_groups.csv.")
        group_rows = list(reader)

    by_id = {}
    members = defaultdict(list)
    for row in group_rows:
        image_id = row["image_id"]
        if image_id in by_id:
            raise ValueError(f"Duplicate group row for image {image_id}.")
        if image_id not in metadata:
            raise ValueError(f"Group row has unknown image {image_id}.")
        if row["label"] != metadata[image_id]["anomaly_class"]:
            raise ValueError(f"Label mismatch for image {image_id}.")
        if not row["group_id"]:
            raise ValueError(f"Missing group ID for image {image_id}.")
        by_id[image_id] = row
        members[row["group_id"]].append(image_id)

    if set(by_id) != set(metadata):
        raise ValueError("Metadata and group file have different image IDs.")

    paths = set()
    for image_id, details in metadata.items():
        relative = Path(details["image_filepath"])
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != "images"
            or ".." in relative.parts
        ):
            raise ValueError(f"Unsafe image path for image {image_id}.")
        if details["image_filepath"] in paths:
            raise ValueError(f"Repeated image path: {relative}")
        paths.add(details["image_filepath"])
        if not (IMAGE_ROOT / relative).is_file():
            raise ValueError(f"Missing image: {relative}")

    for group_id, image_ids in members.items():
        labels = {metadata[image_id]["anomaly_class"] for image_id in image_ids}
        if len(labels) != 1:
            raise ValueError(f"Group {group_id} contains different labels.")
        if any(int(by_id[image_id]["group_size"]) != len(image_ids)
               for image_id in image_ids):
            raise ValueError(f"Incorrect group size for group {group_id}.")

    return metadata, by_id, members


def choose_groups(group_sizes, target, rng):
    """Pick whole groups close to a target count, preserving random ties."""
    shuffled = list(group_sizes)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda item: -item[1])
    chosen = set()
    count = 0
    for group_id, size in shuffled:
        if abs(target - (count + size)) < abs(target - count):
            chosen.add(group_id)
            count += size
    return chosen


def build_assignments(metadata, members, seed):
    by_label = defaultdict(list)
    for group_id, image_ids in members.items():
        label = metadata[image_ids[0]]["anomaly_class"]
        by_label[label].append((group_id, len(image_ids)))

    rng = random.Random(seed)
    assignment = {}
    for label in sorted(by_label):
        groups = by_label[label]
        if len(groups) < 3:
            raise ValueError(f"Class {label} has fewer than three groups.")
        image_count = sum(size for _, size in groups)
        test_groups = choose_groups(groups, round(image_count * TEST_FRACTION), rng)
        remaining = [item for item in groups if item[0] not in test_groups]
        val_groups = choose_groups(remaining, round(image_count * VAL_FRACTION), rng)
        for group_id, _ in groups:
            assignment[group_id] = (
                "test" if group_id in test_groups
                else "val" if group_id in val_groups
                else "train"
            )
        if {assignment[group_id] for group_id, _ in groups} != set(SPLITS):
            raise ValueError(f"Class {label} is missing a split.")
    return assignment


def make_rows(metadata, group_rows, assignment):
    rows = []
    for image_id in sorted(metadata, key=int):
        details = metadata[image_id]
        group_id = group_rows[image_id]["group_id"]
        rows.append({
            "image_id": image_id,
            "image_path": details["image_filepath"],
            "label": details["anomaly_class"],
            "group_id": group_id,
            "split": assignment[group_id],
        })
    return rows


def validate_rows(rows, metadata, group_rows):
    if len(rows) != len(metadata):
        raise ValueError("Manifest row count differs from metadata.")
    seen_ids = set()
    group_split = {}
    class_split = Counter()
    for row in rows:
        image_id = row["image_id"]
        if image_id in seen_ids or image_id not in metadata:
            raise ValueError(f"Duplicate or unknown image ID: {image_id}")
        seen_ids.add(image_id)
        details = metadata[image_id]
        expected_path = details["image_filepath"]
        if (row["image_path"] != expected_path
                or row["label"] != details["anomaly_class"]
                or row["group_id"] != group_rows[image_id]["group_id"]
                or row["split"] not in SPLITS):
            raise ValueError(f"Invalid manifest row for image {image_id}.")
        group_id = row["group_id"]
        if group_id in group_split and group_split[group_id] != row["split"]:
            raise ValueError(f"Group {group_id} crosses split boundaries.")
        group_split[group_id] = row["split"]
        class_split[(row["label"], row["split"])] += 1
    if seen_ids != set(metadata):
        raise ValueError("Manifest omits some images.")
    for label in {details["anomaly_class"] for details in metadata.values()}:
        if any(class_split[(label, split)] == 0 for split in SPLITS):
            raise ValueError(f"Class {label} is absent from a split.")
    return class_split


def image_tree_digest(metadata):
    """Fingerprint image content in stable ID order for team verification."""
    digest = hashlib.sha256()
    for image_id in sorted(metadata, key=int):
        details = metadata[image_id]
        image_hash = sha256_file(IMAGE_ROOT / details["image_filepath"])
        entry = (f"{image_id}\0{details['image_filepath']}\0"
                 f"{details['anomaly_class']}\0{image_hash}\n")
        digest.update(entry.encode("utf-8"))
    return digest.hexdigest()


def print_counts(class_split):
    print("Class                 train    val   test")
    for label in sorted({label for label, _ in class_split}):
        counts = [class_split[(label, split)] for split in SPLITS]
        print(f"{label:20} {counts[0]:5} {counts[1]:6} {counts[2]:6}")
    totals = [sum(count for (label, part), count in class_split.items()
                  if part == split) for split in SPLITS]
    print(f"{'TOTAL':20} {totals[0]:5} {totals[1]:6} {totals[2]:6}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2112)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    summary_path = output.with_suffix(".json")
    metadata, group_rows, members = load_inputs()

    if args.verify_only:
        with output.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != list(FIELDS):
                raise ValueError("Unexpected manifest columns.")
            rows = list(reader)
        class_split = validate_rows(rows, metadata, group_rows)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "manifest_sha256": sha256_file(output),
            "metadata_sha256": sha256_file(METADATA_PATH),
            "groups_sha256": sha256_file(GROUPS_PATH),
            "image_tree_sha256": image_tree_digest(metadata),
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                raise ValueError(f"Dataset or manifest has changed: {key}")
        print("Manifest, groups, metadata, and all image bytes verified.")
    else:
        if output.exists() or summary_path.exists():
            raise SystemExit(
                "Split already exists. Keep it frozen; choose a new output name "
                "for a new evaluation cycle."
            )
        assignment = build_assignments(metadata, members, args.seed)
        rows = make_rows(metadata, group_rows, assignment)
        class_split = validate_rows(rows, metadata, group_rows)
        summary = {
            "seed": args.seed,
            "test_fraction_target": TEST_FRACTION,
            "val_fraction_target": VAL_FRACTION,
            "images": len(rows),
            "groups": len(members),
            "source": "https://github.com/RaptorMaps/InfraredSolarModules",
            "split_limit": (
                "Duplicate-aware stratified benchmark; no panel, site, or "
                "capture IDs are available, so unseen-site generalization "
                "is not established."
            ),
            "metadata_sha256": sha256_file(METADATA_PATH),
            "groups_sha256": sha256_file(GROUPS_PATH),
            "image_tree_sha256": image_tree_digest(metadata),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summary["manifest_sha256"] = sha256_file(output)
        with summary_path.open("x", encoding="utf-8") as target:
            json.dump(summary, target, indent=2)
            target.write("\n")
        print(f"Created frozen manifest: {output}")
        print(f"Verification summary: {summary_path}")

    print_counts(class_split)
    print("Keep the test assignments unchanged and use validation for model selection.")


if __name__ == "__main__":
    main()
