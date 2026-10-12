"""Review similar PV images and create a cross-label-filtered dataset."""

import csv
import html
import json
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "module_metadata.json"
CSV_PATH = PROCESSED_DIR / "near_duplicate_candidates.csv"
HTML_PATH = PROCESSED_DIR / "near_duplicate_review.html"
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

    image_path = PROCESSED_DIR / relative_path
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

cards = []

for candidate in candidates:
    first_path = html.escape(candidate["first_path"], quote=True)
    second_path = html.escape(candidate["second_path"], quote=True)
    first_id = html.escape(candidate["first_id"])
    second_id = html.escape(candidate["second_id"])
    first_label = html.escape(candidate["first_label"])
    second_label = html.escape(candidate["second_label"])
    label_note = "Different labels" if candidate["different_labels"] else "Same label"

    cards.append(
        f"""
        <article>
          <div class="heading">
            <strong>{first_id} and {second_id}</strong>
            <span>{label_note} · hash distance {candidate["hash_distance"]}
            · full-image difference {candidate["full_mae"]:.2f}</span>
          </div>
          <div class="images">
            <figure>
              <img src="{first_path}" alt="Image {first_id}">
              <figcaption>{first_id}: {first_label}</figcaption>
            </figure>
            <figure>
              <img src="{second_path}" alt="Image {second_id}">
              <figcaption>{second_id}: {second_label}</figcaption>
            </figure>
          </div>
        </article>
        """
    )

page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Near-duplicate PV image review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px;
           margin: 2rem auto; padding: 0 1rem; color: #222; }}
    p {{ line-height: 1.5; }}
    article {{ border: 1px solid #ccc; border-radius: 8px;
              padding: 1rem; margin: 1rem 0; }}
    .heading {{ display: flex; justify-content: space-between; gap: 1rem; }}
    .images {{ display: flex; gap: 2rem; }}
    figure {{ margin: 1rem 0 0; }}
    img {{ width: 120px; height: 200px; image-rendering: pixelated;
           border: 1px solid #888; }}
    figcaption {{ margin-top: 0.3rem; }}
  </style>
</head>
<body>
  <h1>Near-duplicate PV image review</h1>
  <p>{len(candidates)} candidate pairs from {len(records)} retained images.
     Pairs have a difference-hash distance of 0 or 1 and an 8×8 thumbnail
     mean absolute difference of at most {MAX_THUMBNAIL_MAE}.
     This is a screening rule, not proof that two images show the same module.
     The original processed images remain unchanged. Both images in every
     different-label candidate pair are excluded from the near-clean dataset.
     Same-label candidates remain and must be kept together in the data split.
     Different-label pairs appear first.</p>
  {''.join(cards)}
</body>
</html>
"""
HTML_PATH.write_text(page, encoding="utf-8")

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
near_clean_images_dir = NEAR_CLEAN_DIR / "images"
expected_paths = {
    records[image_id]["path"] for image_id in near_clean_metadata
}

if len(expected_paths) != len(near_clean_metadata):
    raise SystemExit("Two retained image IDs point to the same image path.")

if near_clean_images_dir.exists():
    for existing_path in near_clean_images_dir.rglob("*"):
        if existing_path.is_file() and existing_path.name != ".DS_Store":
            relative_path = existing_path.relative_to(NEAR_CLEAN_DIR).as_posix()
            if relative_path not in expected_paths:
                raise SystemExit(
                    f"Unexpected old file in near-clean images: {relative_path}"
                )

copied_count = 0
verified_count = 0

for image_id in near_clean_metadata:
    relative_path = Path(records[image_id]["path"])
    source_path = PROCESSED_DIR / relative_path
    destination_path = NEAR_CLEAN_DIR / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        if source_path.read_bytes() != destination_path.read_bytes():
            raise SystemExit(
                f"Near-clean image differs from source: {relative_path}"
            )
        verified_count += 1
    else:
        shutil.copy2(source_path, destination_path)
        copied_count += 1

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
print(f"Images copied this run: {copied_count}")
print(f"Existing copies verified: {verified_count}")
print(f"CSV report: {CSV_PATH}")
print(f"Side-by-side review page: {HTML_PATH}")
print(f"Near-clean metadata: {near_clean_metadata_path}")
print(f"Near-clean exclusions: {exclusions_path}")
print(f"Related-image groups for splitting: {groups_path}")
print("Original raw and exact-clean images were not changed.")
