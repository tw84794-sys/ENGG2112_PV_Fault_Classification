"""List visually similar PV images for human review without excluding them."""

import csv
import html
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "module_metadata.json"
CSV_PATH = PROCESSED_DIR / "near_duplicate_candidates.csv"
HTML_PATH = PROCESSED_DIR / "near_duplicate_review.html"

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
     No images were excluded. Different-label pairs appear first.</p>
  {''.join(cards)}
</body>
</html>
"""
HTML_PATH.write_text(page, encoding="utf-8")

print(f"Processed images checked: {len(records)}")
print(f"Pairs screened by difference hash: {screened_pairs}")
print(f"Pairs selected for review: {len(candidates)}")
print(f"Selected pairs with different labels: {sum(c['different_labels'] for c in candidates)}")
print(f"CSV report: {CSV_PATH}")
print(f"Side-by-side review page: {HTML_PATH}")
print("No images or labels were changed.")
