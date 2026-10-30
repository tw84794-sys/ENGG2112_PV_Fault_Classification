# Current experiment recipes

Each JSON records the exact settings, input fingerprints, run ID and compute limits for a retained reference. Start with the notebooks for the reasoning and results.

| Recipe | Command | Purpose |
|---|---|---|
| [SmallCNN: fewer missed faults](experiment-plan-smallcnn-search-bn-depth.json) | `scripts/train_baseline.py --plan configs/experiment-plan-smallcnn-search-bn-depth.json` | Retained fault-recall reference |
| [SmallCNN: higher macro-F1](experiment-plan-smallcnn-depth-gentle60.json) | `scripts/train_baseline.py` | Default CNN reference |
| [Frozen ResNet parent](experiment-plan-resnet18-frozen-repro.json) | `scripts/train_resnet18.py` | Shared parent for the retained ResNet comparisons |
| [Standard augmented ResNet](experiment-plan-resnet18-generalisation-augmentation.json) | `scripts/finetune_resnet18.py` | Lower-cost ResNet reference |
| [Stride 1](experiment-plan-resnet18-spatial-stride1.json) | `scripts/finetune_resnet18.py --plan configs/experiment-plan-resnet18-spatial-stride1.json` | Highest recorded macro-F1 candidate |
| [No max pooling](experiment-plan-resnet18-spatial-no_maxpool.json) | `scripts/finetune_resnet18.py --plan configs/experiment-plan-resnet18-spatial-no_maxpool.json` | Different missed-fault/false-alarm trade-off |

Prefix commands with `PYTHONPATH=src python` from the repository root. They run short diagnostics unless `--full` is supplied. Existing run IDs are protected. New fitting needs a copied plan with a new run ID; a new parent checkpoint also needs its actual checksum in the child plan.

All other original recipes are retained unchanged in [archive/configs](../archive/configs/). Training commands still resolve their recorded `configs/<filename>.json` paths there, preserving old plan hashes and references. The default recipes above use locally available retained parents; the old CV-selected recipe requires additional historical artifacts.

The [final-evaluation protocol](final-evaluation-v1.json) records the selected checkpoints, validation-only screening cutoffs, environment and code fingerprints before the reserved test pass. It is an evaluation record, not another training recipe.
