"""Data-audit entry point (not implemented).

Check image decoding, shape/channel, labels, imbalance, contradictory annotations,
exact/near duplicates and available independent-unit identifiers before splitting.
"""

import json
from collections import Counter
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

for image_id, details in metadata.items():
    label = details["anomaly_class"]
    image_path = raw_dir / details["image_filepath"]

    labels[label] += 1

    try:
        with Image.open(image_path) as image:
            image.load()  # Actually read the image, rather than just its filename.
            sizes[image.size] += 1
            modes[image.mode] += 1
    except OSError as error:
        problems.append((image_id, str(image_path), str(error)))

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

for image_id, path, error in problems:
    print(f"Problem with image {image_id}: {path} — {error}")