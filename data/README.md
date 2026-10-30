# Dataset and shared split

This project uses [Raptor Maps' InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules): 20,000 grayscale, 24 × 40-pixel module images with 12 class labels. The source repository includes an [MIT license](https://github.com/RaptorMaps/InfraredSolarModules/blob/master/LICENSE); retain its copyright and permission notice when redistributing the data.

Download `2020-02-14_InfraredSolarModules.zip` from the source repository and extract it so these paths exist:

```text
data/raw/images/0.jpg
data/raw/module_metadata.json
```

The image folders are ignored by Git. Each teammate creates the same local cleaning records from the source:

```bash
python scripts/audit_data.py
python scripts/review_near_duplicates.py
python scripts/create_split.py --verify-only
```

The first script excludes exact duplicates and conflicting exact-label pairs. The second excludes both images in each different-label near-duplicate candidate pair and records groups of retained same-label candidates. The last command checks the tracked split against the local metadata, groups, and every image's bytes. If it fails, do not train until the dataset versions and cleaning scripts match.

The tracked split is `data/splits/infrared_solar_modules_v1.csv`, with its seed and fingerprints in the adjacent JSON file. In the CSV, `image_path` is relative to `data/raw/`; for example, `images/0.jpg` points to `data/raw/images/0.jpg`. Training code should select rows by the `split` column and keep all rows with the same `group_id` in one partition. Do not rerun split creation or alter test assignments after model selection starts. A changed cleaning rule needs a new manifest version and evaluation cycle.

This is a duplicate-aware benchmark. Source metadata has no verified panel, site, or capture identifiers, and the near-duplicate rule is only a visual screen. The split cannot establish performance on unseen panels or sites.

Cleaning writes filtered metadata, exclusions and related-image groups under `data/processed/`; it does not copy image files. Existing older processed image copies may remain locally, but current code does not use them. Keep the original files under `data/raw/`. The frozen split and fingerprints are unchanged.

## Loading batches

With the project dependencies installed, run Python with `PYTHONPATH=src`:

```python
from pv_fault.data import CLASS_NAMES, make_data_loaders

loaders = make_data_loaders(batch_size=64, seed=2112)
images, labels = next(iter(loaders["train"]))
# images: float32 [batch, 1, 40, 24], brightness scaled to [0, 1]
# labels: int64 [batch]; CLASS_NAMES[index] gives the original label
```

The loader preserves the frozen assignments, verifies manifest and metadata
fingerprints, and checks label consistency and group separation. Run the full
`--verify-only` command above before experiments to check every image's bytes.
Images remain grayscale at their native size. Only training is shuffled; all
examples, including the last partial batch, are retained. Optional
`train_transform` receives a tensor and applies only to training images. No
augmentation is enabled by default. The loader fits no preprocessing statistics;
any future learned normalization must use training data only.

`num_workers=0` is the default for notebooks and macOS. When using multiple
workers in a script, create loaders inside an `if __name__ == "__main__":` guard.
The seed controls batch ordering and worker random generators. Seed Python,
NumPy and PyTorch in the training entry point as well when using random transforms
with zero workers.

A custom `data_root` passed to the loader is the folder containing `raw/` and `processed/`, matching the repository’s `data/` layout.
