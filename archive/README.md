# Experiment history

The main notebooks present the retained workflow and its key decisions. This archive keeps the older recipes and tools available without placing every trial in the main reading path. All original recipe JSON bytes are unchanged; result JSON, Markdown records, figures and frozen data splits remain at their original paths under `results/` and `data/splits/`.

## Detailed evidence

| Question | Record |
|---|---|
| Width, learning rate and training duration | [CNN comparison](../results/cnn-training-comparison-v1.md) |
| Scheduler and BatchNorm controls | [CNN ablation](../results/cnn-scheduler-batchnorm-v1.md) |
| AdamW and SGD follow-up | [Optimisers](../results/cnn-optimizers-v1.md) |
| GELU, longer training and augmentation | [SmallCNN follow-up](../results/smallcnn-20min-v1.md) |
| Selected CNN errors | [Error review](../results/cnn-error-review-v1.md) |
| Original frozen ResNet | [Frozen validation record](../results/resnet18-frozen-v1-validation.md) |
| Fresh frozen and layer4 reproduction | [Reproduction](../results/resnet18-reproduction-v1.md) |
| ResNet optimisation and compute | [Bounded follow-up](../results/resnet18-hour-v1.md) |
| Augmentation, smoothing, weighting and staged adaptation | [Generalisation](../results/resnet18-generalisation-v1.md) |
| Early spatial downsampling | [Spatial comparison](../results/resnet18-spatial-v1.md) |

The [results directory](../results/) also retains search snapshots, ledgers, original histories, complete metrics and verification records, including unsuccessful and provisional trials. These records distinguish validation selection from test performance.

## Archived commands

Run commands from the repository root with `PYTHONPATH=src python archive/scripts/<filename>`. Scripts use the current shared library and their original validation/overwrite safeguards. Training defaults to diagnostics; full reproduction requires the recorded data, pinned weights, parent artifacts and an unused run ID.

| Tool | Purpose |
|---|---|
| [create_cv_folds.py](scripts/create_cv_folds.py) | Recreate or verify training-only three-fold assignments |
| [finetune_resnet18_cv.py](scripts/finetune_resnet18_cv.py) | Training-only learning-rate comparison |
| [finetune_resnet18_selected.py](scripts/finetune_resnet18_selected.py) | Compatibility command for the original CV-selected refit |
| [check_cnn_overfit.py](scripts/check_cnn_overfit.py) | Diagnostic fitting on a tiny training subset |
| [analyze_normal_routing.py](scripts/analyze_normal_routing.py) | Exploratory routing thresholds from saved validation predictions |
| [evaluate_external_suas.py](scripts/evaluate_external_suas.py) | Exploratory binary projection on the separate archive |
| [review_resnet_errors.py](scripts/review_resnet_errors.py) | Export errors for the original fixed run |
| [benchmark_cnn_resnet.py](scripts/benchmark_cnn_resnet.py) | Synthetic checkpoint inference timing |
| [benchmark_resnet_spatial.py](scripts/benchmark_resnet_spatial.py) | Matched spatial-variant timing |
| [summarise_resnet_generalisation.py](scripts/summarise_resnet_generalisation.py) | Export verified generalisation results |
| [summarise_resnet_spatial.py](scripts/summarise_resnet_spatial.py) | Export verified spatial results |

Some reporting tools explicitly regenerate their result files when run; do not use them merely to read results. The retained snapshots can be read without local checkpoints. Raw data, external archives and full run folders are not distributed through Git.

## Original recipes and provenance

Older recipes and protocol documents are in [configs/](configs/). Current training commands resolve a missing original `configs/<filename>.json` to `archive/configs/<filename>.json`. Explicit archived paths also work. Saved JSON strings and their hashes are not rewritten to implement this move.

Historical checkpoint verification keeps checking archived producer hashes and compares all saved predictions/probabilities with current inference. Refactored live-source hashes are recorded separately. Reproduction with current code records a new producer; it does not rewrite the original run.

The original CV-selected recipe expects a frozen-parent path that differs from the later reproduction folder. The later frozen checkpoint has the exact expected checksum, but the original CV summary is missing locally. That older recipe is not fully runnable from this checkout without recovering the summary. Current ResNet recipes use the retained frozen reproduction checkpoint directly.

The longer original notebooks remain available in Git history: [CNN development log](https://github.com/tw84794-sys/ENGG2112_PV_Fault_Classification/blob/0030d696d0be1329eb80f1ffa624e88a3938728f/notebooks/02_baseline_experiments.ipynb) and [ResNet development log](https://github.com/tw84794-sys/ENGG2112_PV_Fault_Classification/blob/0030d696d0be1329eb80f1ffa624e88a3938728f/notebooks/03_resnet18_experiments.ipynb).
