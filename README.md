# ENGG2112: thermal solar-panel fault classification

We compare compact CNNs and pretrained ResNet18 on 12 classes of thermal solar-module images. The goal is to support further inspection and reuse assessment. Predictions do not certify panel safety or suitability for reuse.

## Read the project

1. [Data exploration](notebooks/01_data_exploration.ipynb): image properties, imbalance, cleaning and frozen splits.
2. [CNN baseline](notebooks/02_baseline_experiments.ipynb): model design, useful comparisons, two retained references and errors.
3. [ResNet experiments](notebooks/03_resnet18_experiments.ipynb): transfer learning, controlled spatial variants, remaining errors and compute cost.

The experiment notebooks display recorded results without training. Their image examples are optional; notebook 01 requires the local dataset. Executed plots remain visible on GitHub.

## Run locally

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Launch from the repository root. Follow [data setup](data/README.md) before loading images or fitting models. The six [current recipes](configs/README.md) describe the retained CNN references, frozen ResNet parent and three ResNet spatial variants.

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python scripts/create_split.py --verify-only
PYTHONPATH=src python scripts/train_baseline.py
PYTHONPATH=src python scripts/fetch_resnet18_weights.py --verify-only
PYTHONPATH=src python scripts/train_resnet18.py
PYTHONPATH=src python scripts/finetune_resnet18.py
```

Training commands default to two-batch diagnostics. Full fitting requires `--full`, a new run ID in a copied plan, and the exact recorded inputs/parent checkpoint. Existing full runs cannot be overwritten. Download missing ImageNet weights with `PYTHONPATH=src python scripts/fetch_resnet18_weights.py` before ResNet diagnostics.

## Evidence and history

- [CNN error review](results/cnn-error-review-v1.md): higher macro-F1 versus fewer missed faults.
- [ResNet spatial comparison](results/resnet18-spatial-v1.md): performance, feature geometry and actual compute measurements.
- [Experiment archive](archive/README.md): older recipes, reporting tools and detailed trial records.

Historical JSON records, figures, frozen splits and archived producer sources are preserved. Shared code handles setup, source snapshots, fitting, validation and plotting. Current training uses one validation pass per epoch and verifies every saved probability after reloading the selected checkpoint; older reported timings describe their original producers.

These are mainly single-seed, validation-selected experiments. The duplicate-aware split has no verified site/panel identifiers; it cannot establish unseen-site performance. Original-test inference remains reserved. Raw data, checkpoints, environments and full local runs stay outside Git.

## Final reserved-test evaluation

`scripts/evaluate_final.py` freezes the majority baseline, selected SmallCNN and standard augmented ResNet. It reproduces saved validation probabilities and selects each learned model's binary screening threshold on validation only: the highest cutoff achieving at least 90% anomaly recall, with the resulting false-alarm rate checked against 20%. Infeasible targets are recorded honestly.

```bash
PYTHONPATH=src python scripts/evaluate_final.py freeze
```

Review and commit `configs/final-evaluation-v1.json` together with its evaluation code before running:

```bash
PYTHONPATH=src python scripts/evaluate_final.py test
```

The test command requires the committed protocol, unchanged code/environment, matching checkpoint fingerprints and the verified dataset. It records a start marker before loading test images and refuses to overwrite any started or completed evaluation. If execution fails after that marker, preserve the partial folder and review what was exposed before continuing; do not delete it to rerun model selection.

Multiclass argmax predictions and threshold-based anomaly screening are scored separately. Results include per-class metrics, confusion matrices and seeded 95% bootstrap intervals resampling detected related-image groups. Group intervals do not establish independent-site generalisation. Compact results and figures go under `results/final-evaluation-v1*`; full prediction vectors stay in ignored `runs/final-evaluation-v1/`.
