"""Data-audit entry point (not implemented).

Check image decoding, shape/channel, labels, imbalance, contradictory annotations,
exact/near duplicates and available independent-unit identifiers before splitting.
"""

import csv
import json
import shutil
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from PIL import Image


# Find data/raw relative to this script, wherever the project is stored.
project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
metadata_path = raw_dir / "module_metadata.json"
processed_dir = project_dir / "data" / "processed"

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

labels = Counter()
sizes = Counter()
modes = Counter()
problems = []
image_groups = defaultdict(list)


def checked_relative_path(path_string):
    relative_path = Path(path_string)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or relative_path.parts[0] != "images"
        or ".." in relative_path.parts
    ):
        raise ValueError(f"Unexpected image path in metadata: {path_string}")
    return relative_path


for image_id, details in metadata.items():
    label = details["anomaly_class"]
    relative_path = checked_relative_path(details["image_filepath"])
    image_path = raw_dir / relative_path

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
                (image_id, relative_path.as_posix(), label)
            )

    except OSError as error:
        problems.append((image_id, str(image_path), str(error)))

if problems:
    for image_id, path, error in problems:
        print(f"Problem with image {image_id}: {path} — {error}")
    raise SystemExit("Resolve unreadable images before building processed data.")

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

for _, images in duplicate_groups:
    with Image.open(raw_dir / images[0][1]) as image:
        image.load()
        reference = (image.mode, image.size, image.tobytes())

    for image_id, relative_path, _ in images[1:]:
        with Image.open(raw_dir / relative_path) as image:
            image.load()
            candidate = (image.mode, image.size, image.tobytes())
        if candidate != reference:
            raise SystemExit(
                f"Fingerprint match failed direct pixel check: {image_id}"
            )

exclusions = {}

for group_number, (_, images) in enumerate(duplicate_groups, start=1):
    group_labels = {label for _, _, label in images}

    if len(group_labels) > 1:
        for image_id, _, _ in images:
            exclusions[image_id] = ("conflicting_labels", group_number, "")
    else:
        kept_image_id = min(images, key=lambda row: int(row[0]))[0]
        for image_id, _, _ in images:
            if image_id != kept_image_id:
                exclusions[image_id] = (
                    "exact_duplicate",
                    group_number,
                    kept_image_id,
                )

clean_metadata = {
    image_id: details
    for image_id, details in metadata.items()
    if image_id not in exclusions
}

processed_dir.mkdir(parents=True, exist_ok=True)
processed_images_dir = processed_dir / "images"

expected_paths = {
    checked_relative_path(details["image_filepath"]).as_posix()
    for details in clean_metadata.values()
}

if processed_images_dir.exists():
    for existing_path in processed_images_dir.rglob("*"):
        if existing_path.is_file() and existing_path.name != ".DS_Store":
            relative_path = existing_path.relative_to(processed_dir).as_posix()
            if relative_path not in expected_paths:
                raise SystemExit(
                    f"Unexpected old file in processed images: {relative_path}"
                )

copied_count = 0
verified_count = 0

for details in clean_metadata.values():
    relative_path = checked_relative_path(details["image_filepath"])
    source_path = raw_dir / relative_path
    destination_path = processed_dir / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        if source_path.read_bytes() != destination_path.read_bytes():
            raise SystemExit(
                f"Processed image differs from raw image: {relative_path}"
            )
        verified_count += 1
    else:
        shutil.copy2(source_path, destination_path)
        copied_count += 1

duplicate_report_path = processed_dir / "duplicate_groups.csv"

with duplicate_report_path.open("w", newline="", encoding="utf-8") as report_file:
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

exclusions_path = processed_dir / "exclusions.csv"

with exclusions_path.open("w", newline="", encoding="utf-8") as report_file:
    writer = csv.writer(report_file)
    writer.writerow(
        ["image_id", "image_path", "label", "reason", "group_number", "kept_image_id"]
    )

    for image_id in sorted(exclusions, key=int):
        reason, group_number, kept_image_id = exclusions[image_id]
        details = metadata[image_id]
        writer.writerow(
            [
                image_id,
                details["image_filepath"],
                details["anomaly_class"],
                reason,
                group_number,
                kept_image_id,
            ]
        )

clean_metadata_path = processed_dir / "module_metadata.json"
clean_metadata_path.write_text(
    json.dumps(clean_metadata, indent=2) + "\n",
    encoding="utf-8",
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
print(f"Images excluded: {len(exclusions)}")
print(f"Clean images retained: {len(clean_metadata)}")
print(f"Images copied this run: {copied_count}")
print(f"Existing copies verified: {verified_count}")
print(f"Duplicate report: {duplicate_report_path}")
print(f"Exclusion report: {exclusions_path}")
print(f"Clean metadata: {clean_metadata_path}")
