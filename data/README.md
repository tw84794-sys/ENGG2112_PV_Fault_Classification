# Dataset and shared split

This project uses [Raptor Maps' InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules): 20,000 grayscale, 24 × 40-pixel module images with 12 class labels. The source repository includes an [MIT license](https://github.com/RaptorMaps/InfraredSolarModules/blob/master/LICENSE); retain its copyright and permission notice when redistributing the data.

Download `2020-02-14_InfraredSolarModules.zip` from the source repository and extract it so these paths exist:

```text
data/raw/images/0.jpg
data/raw/module_metadata.json
```

The image folders are ignored by Git. Each teammate creates the same local cleaned dataset from the source:

```bash
python scripts/audit_data.py
python scripts/review_near_duplicates.py
python scripts/create_split.py --verify-only
```

The first script excludes exact duplicates and conflicting exact-label pairs. The second excludes both images in each different-label near-duplicate candidate pair and records groups of retained same-label candidates. The last command checks the tracked split against the local metadata, groups, and every image's bytes. If it fails, do not train until the dataset versions and cleaning scripts match.

The tracked split is `data/splits/infrared_solar_modules_v1.csv`, with its seed and fingerprints in the adjacent JSON file. In the CSV, `image_path` is relative to `data/processed/near_clean/`; for example, `images/0.jpg` points to `data/processed/near_clean/images/0.jpg`. Training code should select rows by the `split` column and keep all rows with the same `group_id` in one partition. Do not rerun split creation or alter test assignments after model selection starts. A changed cleaning rule needs a new manifest version and evaluation cycle.

This is a duplicate-aware benchmark. Source metadata has no verified panel, site, or capture identifiers, and the near-duplicate rule is only a visual screen. The split cannot establish performance on unseen panels or sites.
