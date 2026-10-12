"""Data-audit entry point (not implemented).

Check image decoding, shape/channel, labels, imbalance, contradictory annotations,
exact/near duplicates and available independent-unit identifiers before splitting.
"""

import csv
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from PIL import Image


# Find data/raw relative to this script, wherever the project is stored.
project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
metadata_path = raw_dir / "module_metadata.json"

metadata = json.loads(metadata_path.read_text())

labels = Counter()
sizes = Counter()
modes = Counter()
problems = []
image_groups = defaultdict(list)

for image_id, details in metadata.items():
    label = details["anomaly_class"]
    image_path = raw_dir / details["image_filepath"]

    labels[label] += 1

    try:
        with Image.open(image_path) as image:
            image.load()  # Actually read the image, rather than just its filename.
            sizes[image.size] += 1
            modes[image.mode] += 1

            fingerprint_data = (
                image.mode.encode("ascii")
                + b"|"
                + str(image.size).encode("ascii")
                + b"|"
                + image.tobytes()
            )
            fingerprint = sha256(fingerprint_data).hexdigest()

            image_groups[fingerprint].append(
                (image_id, details["image_filepath"], label)
            )

    except OSError as error:
        problems.append((image_id, str(image_path), str(error)))

duplicate_groups = [
    (fingerprint, images)
    for fingerprint, images in sorted(image_groups.items())
    if len(images) > 1
]

conflicting_groups = [
    images
    for _, images in duplicate_groups
    if len({label for _, _, label in images}) > 1
]

report_dir = project_dir / "data" / "processed"
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / "duplicate_groups.csv"

with report_path.open("w", newline="", encoding="utf-8") as report_file:
    writer = csv.writer(report_file)
    writer.writerow(
        ["group_number", "image_id", "image_path", "label", "label_conflict"]
    )

    for group_number, (_, images) in enumerate(duplicate_groups, start=1):
        has_conflict = len({label for _, _, label in images}) > 1

        for image_id, relative_path, label in images:
            writer.writerow(
                [group_number, image_id, relative_path, label, has_conflict]
            )

print(f"Images listed in metadata: {len(metadata)}")
print(f"Images opened successfully: {sum(sizes.values())}")
print(f"Images with problems: {len(problems)}")

print("\nLabels:")
for label, count in sorted(labels.items()):
    print(f"  {label}: {count}")

print("\nImage sizes (width, height):")
for size, count in sorted(sizes.items()):
    print(f"  {size}: {count}")

print("\nColour modes:")
for mode, count in sorted(modes.items()):
    print(f"  {mode}: {count}")

print(f"\nGroups of identical images: {len(duplicate_groups)}")
print(f"Images in those groups: {sum(len(images) for _, images in duplicate_groups)}")
print(f"Groups with conflicting labels: {len(conflicting_groups)}")
print(f"Duplicate report: {report_path}")

for image_id, path, error in problems:
    print(f"Problem with image {image_id}: {path} — {error}")