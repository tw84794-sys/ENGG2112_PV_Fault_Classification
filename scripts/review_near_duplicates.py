"""Review similar PV images and create a cross-label-filtered dataset."""

import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "module_metadata.json"
CSV_PATH = PROCESSED_DIR / "near_duplicate_candidates.csv"
NEAR_CLEAN_DIR = PROCESSED_DIR / "near_clean"

MAX_THUMBNAIL_MAE = 4.0
EXPECTED_SIZE = (24, 40)


def difference_hash(image):
    resized = image.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = resized.tobytes()
    result = 0

    for y in range(8):
        for x in range(8):
            left = pixels[y * 9 + x]
            right = pixels[y * 9 + x + 1]
            result = (result << 1) | int(left > right)

    return result


def mean_absolute_error(first, second):
    if len(first) != len(second):
        raise ValueError("Images being compared have different pixel counts.")

    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def numeric_id(image_id):
    return int(image_id)


metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
records = {}
hash_buckets = defaultdict(list)

for image_id, details in metadata.items():
    relative_path = Path(details["image_filepath"])
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or relative_path.parts[0] != "images"
        or ".." in relative_path.parts
    ):
        raise ValueError(f"Unexpected image path: {relative_path}")

    image_path = PROJECT_DIR / "data/raw" / relative_path
    with Image.open(image_path) as image:
        image.load()
        if image.mode != "L" or image.size != EXPECTED_SIZE:
            raise ValueError(
                f"Unexpected mode or size for {image_path}: "
                f"{image.mode}, {image.size}"
            )

        fingerprint = difference_hash(image)
        thumbnail = image.resize((8, 8), Image.Resampling.LANCZOS).tobytes()
        full_pixels = image.tobytes()

    records[image_id] = {
        "label": details["anomaly_class"],
        "path": relative_path.as_posix(),
        "thumbnail": thumbnail,
        "full_pixels": full_pixels,
    }
    hash_buckets[fingerprint].append(image_id)

candidates = []


def consider_pair(first_id, second_id, hash_distance):
    first = records[first_id]
    second = records[second_id]
    thumbnail_mae = mean_absolute_error(
        first["thumbnail"], second["thumbnail"]
    )

    if thumbnail_mae > MAX_THUMBNAIL_MAE:
        return

    full_mae = mean_absolute_error(
        first["full_pixels"], second["full_pixels"]
    )
    first_id, second_id = sorted((first_id, second_id), key=numeric_id)
    first = records[first_id]
    second = records[second_id]

    candidates.append(
        {
            "first_id": first_id,
            "first_path": first["path"],
            "first_label": first["label"],
            "second_id": second_id,
            "second_path": second["path"],
            "second_label": second["label"],
            "different_labels": first["label"] != second["label"],
            "hash_distance": hash_distance,
            "thumbnail_mae": thumbnail_mae,
            "full_mae": full_mae,
        }
    )


screened_pairs = 0

for fingerprint, image_ids in hash_buckets.items():
    for first_id, second_id in combinations(image_ids, 2):
        screened_pairs += 1
        consider_pair(first_id, second_id, 0)

    for bit in range(64):
        neighbouring_hash = fingerprint ^ (1 << bit)
        if neighbouring_hash <= fingerprint:
            continue

        for first_id in image_ids:
            for second_id in hash_buckets.get(neighbouring_hash, []):
                screened_pairs += 1
                consider_pair(first_id, second_id, 1)

candidates.sort(
    key=lambda item: (
        not item["different_labels"],
        item["full_mae"],
        numeric_id(item["first_id"]),
        numeric_id(item["second_id"]),
    )
)

with CSV_PATH.open("w", newline="", encoding="utf-8") as output:
    fieldnames = [
        "first_id",
        "first_path",
        "first_label",
        "second_id",
        "second_path",
        "second_label",
        "different_labels",
        "hash_distance",
        "thumbnail_mae",
        "full_mae",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for candidate in candidates:
        writer.writerow(
            {
                **candidate,
                "thumbnail_mae": f'{candidate["thumbnail_mae"]:.3f}',
                "full_mae": f'{candidate["full_mae"]:.3f}',
            }
        )

cross_label_pairs = [
    candidate for candidate in candidates if candidate["different_labels"]
]
related_cross_label_ids = defaultdict(set)

for candidate in cross_label_pairs:
    first_id = candidate["first_id"]
    second_id = candidate["second_id"]
    related_cross_label_ids[first_id].add(second_id)
    related_cross_label_ids[second_id].add(first_id)

excluded_ids = set(related_cross_label_ids)
near_clean_metadata = {
    image_id: details
    for image_id, details in metadata.items()
    if image_id not in excluded_ids
}

NEAR_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
exclusions_path = NEAR_CLEAN_DIR / "exclusions.csv"

with exclusions_path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(
        ["image_id", "image_path", "label", "reason", "paired_image_ids"]
    )

    for image_id in sorted(excluded_ids, key=numeric_id):
        writer.writerow(
            [
                image_id,
                records[image_id]["path"],
                records[image_id]["label"],
                "cross_label_near_duplicate_candidate",
                ";".join(
                    sorted(related_cross_label_ids[image_id], key=numeric_id)
                ),
            ]
        )

parent = {image_id: image_id for image_id in near_clean_metadata}


def find_group(image_id):
    while parent[image_id] != image_id:
        parent[image_id] = parent[parent[image_id]]
        image_id = parent[image_id]
    return image_id


def join_groups(first_id, second_id):
    first_root = find_group(first_id)
    second_root = find_group(second_id)
    if first_root != second_root:
        parent[second_root] = first_root


for candidate in candidates:
    if candidate["different_labels"]:
        continue
    first_id = candidate["first_id"]
    second_id = candidate["second_id"]
    if first_id in parent and second_id in parent:
        join_groups(first_id, second_id)

members_by_root = defaultdict(list)
for image_id in near_clean_metadata:
    members_by_root[find_group(image_id)].append(image_id)

group_info = {}
for member_ids in members_by_root.values():
    group_id = min(member_ids, key=numeric_id)
    for image_id in member_ids:
        group_info[image_id] = (group_id, len(member_ids))

groups_path = NEAR_CLEAN_DIR / "related_groups.csv"
with groups_path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(["image_id", "label", "group_id", "group_size"])
    for image_id in sorted(near_clean_metadata, key=numeric_id):
        group_id, group_size = group_info[image_id]
        writer.writerow(
            [image_id, records[image_id]["label"], group_id, group_size]
        )

near_clean_metadata_path = NEAR_CLEAN_DIR / "module_metadata.json"
near_clean_metadata_path.write_text(
    json.dumps(near_clean_metadata, indent=2) + "\n",
    encoding="utf-8",
)

removed_by_class = Counter(
    records[image_id]["label"] for image_id in excluded_ids
)
multi_image_groups = sum(
    len(member_ids) > 1 for member_ids in members_by_root.values()
)

print(f"Processed images checked: {len(records)}")
print(f"Pairs screened by difference hash: {screened_pairs}")
print(f"Pairs selected for review: {len(candidates)}")
print(f"Selected pairs with different labels: {len(cross_label_pairs)}")
print(f"Distinct cross-label images excluded: {len(excluded_ids)}")
print(f"Near-clean images retained: {len(near_clean_metadata)}")
print(f"Excluded images by class: {dict(sorted(removed_by_class.items()))}")
print(f"Same-label groups with multiple retained images: {multi_image_groups}")
print(f"CSV report: {CSV_PATH}")
print(f"Near-clean metadata: {near_clean_metadata_path}")
print(f"Near-clean exclusions: {exclusions_path}")
print(f"Related-image groups for splitting: {groups_path}")
print("Original raw and exact-clean images were not changed.")
