"""Make three reproducible, group-aware folds from the existing train split only.

This does not move images or alter the original train/val/test manifest.
"""

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = "8c6c80cc05943e85d4a03874b1d5d89cb83eaf98e5637560b9ec9a2dcdee1ef8"
CLASS_NAMES = (
    "Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi", "Hot-Spot",
    "Hot-Spot-Multi", "No-Anomaly", "Offline-Module", "Shadowing",
    "Soiling", "Vegetation",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build(root, seed=2112, folds=3):
    if folds != 3:
        raise ValueError("This protocol uses exactly three folds.")

    manifest = root / "data/splits/infrared_solar_modules_v1.csv"
    source_bytes = manifest.read_bytes()
    if sha256(source_bytes) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Original split manifest changed; stop and audit it.")

    rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"))))
    required = ["image_id", "image_path", "label", "group_id", "split"]
    if not rows or list(rows[0]) != required:
        raise ValueError("Unexpected manifest columns.")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image IDs in original manifest.")

    groups = defaultdict(list)
    reserved_groups = set()
    for row in rows:
        if row["split"] == "train":
            if row["label"] not in CLASS_NAMES or not row["group_id"]:
                raise ValueError("Invalid training label or group.")
            groups[row["group_id"]].append(row)
        elif row["split"] in ("val", "test"):
            reserved_groups.add(row["group_id"])
        else:
            raise ValueError("Unknown source split.")
    if not groups or set(groups) & reserved_groups:
        raise ValueError("Original groups overlap between train and val/test.")

    by_class = defaultdict(list)
    for group_id, members in groups.items():
        labels = {row["label"] for row in members}
        if len(labels) != 1:
            raise ValueError(f"Mixed-label group needs manual review: {group_id}")
        by_class[members[0]["label"]].append((group_id, members))

    rng = random.Random(seed)
    assignments = {}
    class_sizes = [Counter() for _ in range(folds)]
    total_sizes = [0] * folds
    group_sizes = [0] * folds
    for label in CLASS_NAMES:
        members = by_class[label]
        if len(members) < folds:
            raise ValueError(f"Class {label} has fewer groups than folds.")
        rng.shuffle(members)
        members.sort(key=lambda pair: -len(pair[1]))
        for group_id, group_rows in members:
            fold = min(
                range(folds),
                key=lambda index: (
                    class_sizes[index][label], total_sizes[index],
                    group_sizes[index], index,
                ),
            )
            assignments[group_id] = fold
            class_sizes[fold][label] += len(group_rows)
            total_sizes[fold] += len(group_rows)
            group_sizes[fold] += 1

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["image_id", "image_path", "label", "group_id", "fold"])
    for row in rows:
        if row["split"] == "train":
            writer.writerow([
                row["image_id"], row["image_path"], row["label"],
                row["group_id"], assignments[row["group_id"]],
            ])
    fold_bytes = output.getvalue().encode("utf-8")
    summary = {
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "fold_manifest_sha256": sha256(fold_bytes),
        "seed": seed,
        "folds": folds,
        "images": sum(total_sizes),
        "groups": len(groups),
        "held_out_original_val_test": True,
        "fold_summary": [
            {
                "fold": index,
                "images": total_sizes[index],
                "groups": group_sizes[index],
                "class_counts": dict(sorted(class_sizes[index].items())),
            }
            for index in range(folds)
        ],
        "limit": (
            "Duplicate-aware folds within the original training partition. "
            "No panel/site/capture IDs; no unseen-site generalization claim."
        ),
    }
    if any(set(item["class_counts"]) != set(CLASS_NAMES)
           for item in summary["fold_summary"]):
        raise ValueError("A fold is missing a class.")
    summary_bytes = (json.dumps(summary, indent=2) + "\n").encode("utf-8")
    return fold_bytes, summary_bytes, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=2112)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "data/splits/infrared_solar_modules_train_cv3.csv"
    info = output.with_suffix(".json")
    fold_bytes, summary_bytes, summary = build(root, args.seed)
    if args.verify_only:
        if output.read_bytes() != fold_bytes or info.read_bytes() != summary_bytes:
            raise ValueError("Saved folds differ from the source and seed.")
        print("Fold manifest and checksum verified; original val/test untouched.")
    else:
        if output.exists() or info.exists():
            raise FileExistsError("Fold files exist. Use --verify-only; do not overwrite.")
        output.write_bytes(fold_bytes)
        info.write_bytes(summary_bytes)
        print(f"Saved {output} and {info}")
    print(json.dumps({
        "source_manifest_sha256": summary["source_manifest_sha256"],
        "fold_manifest_sha256": summary["fold_manifest_sha256"],
        "fold_images": [item["images"] for item in summary["fold_summary"]],
        "fold_groups": [item["groups"] for item in summary["fold_summary"]],
        "fold_min_class_count": [
            min(item["class_counts"].values()) for item in summary["fold_summary"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
