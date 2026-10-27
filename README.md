# ENGG2112 AI for Good PV_Fault_Classification

Australia’s growing volume of photovoltaic (PV) cells creates a need for efficient methods of identifying panels that may still be suitable for reuse and repair rather than being unnecessarily recycled or discarded. This project proposes an AI-supported screening system that uses traditional machine learning and convolutional neural network (CNN) models to classify common defects in PV cells from thermal imagery. Publicly available labelled faulty and unfaulty PV image datasets will be used to train, validate and test the model, with transfer learning considered to enable a feasible implementation within the project timeframe. Model performance will be assessed using accuracy, precision, recall, F1-score and confusion matrices. Techniques such as Grad_CAM will be used to highlight image regions primarily influencing predictions. Emphasis will be made on reducing false negative classifications for safety-critical defects. The intended goal is a decision support tool that identifies potential panel defects and helps prioritise decommissioned cells for further reuse testing, repair assessments or recycling, whilst leaving final safety and reuse decisions to qualified technicians.


Dataset:

InfraredSolarModules

https://github.com/RaptorMaps/InfraredSolarModules

## Frozen ResNet18 comparison

After recreating the cleaned data and frozen split, activate the project's environment and run these commands from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/fetch_resnet18_weights.py
PYTHONPATH=src .venv/bin/python scripts/fetch_resnet18_weights.py --verify-only
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
PYTHONPATH=src .venv/bin/python scripts/train_resnet18.py --plan configs/experiment-plan-resnet18.json
PYTHONPATH=src .venv/bin/python scripts/train_resnet18.py --plan configs/experiment-plan-resnet18.json --full
```

The first training command runs a two-batch smoke check. The final command runs one bounded trial and refuses to overwrite an existing run. It saves its checkpoint, metrics and validation predictions under `runs/resnet18-frozen-s2112-cpu-v1/`, which is excluded from Git. The [validation comparison](results/resnet18-frozen-v1-validation.md) contains the shareable results and limitations. The frozen test split remains untouched until independent review.

## CNN training comparison

The [extended CNN comparison](results/cnn-training-comparison-v1.md) records four
training/validation-only trials: SmallCNN and WideCNN, each at learning rates
0.001 and 0.0003. The explicit plans allow up to 30 epochs, five epochs of
early-stopping patience, and 300 seconds per trial. Architectures, class weighting,
batch size and seed remain fixed. Each saved checkpoint is reloaded to verify its
validation predictions. Full weights and image-level predictions stay in ignored
`runs/` folders; shareable metrics and learning curves are under `results/`.

## Optimiser follow-up

The [AdamW/SGD follow-up](results/cnn-optimizers-v1.md) compares six new trials on
the matched two-layer BatchNorm CNNs, using seed 2112 without seed checks.
The baseline experiments notebook records the settings, curves, error trade-offs
and comparison with the earlier deeper models. Test images remain reserved.

## Three-layer SmallCNN follow-up

The [GELU and training-duration follow-up](results/smallcnn-20min-v1.md) records
activation, longer-training, mild-augmentation and gentler-scheduler trials.
The highest macro-F1 candidate misses more faults than the previous reference;
the baseline notebook retains both results and the decision trade-off.

### Consolidated baseline error review

The [baseline error review](results/cnn-error-review-v1.md) compares the two SmallCNN references, their weakest classes and common confusions. The baseline notebook includes their confusion matrices and illustrative validation errors. Higher macro-F1 came with more missed faults; retain both references for the separate ResNet comparison.

### ResNet experiment notebook

[ResNet18 experiments](notebooks/03_resnet18_experiments.ipynb) explains the existing frozen-backbone, layer4 fine-tuning, augmentation and learning-rate progression, then compares it with the optimised SmallCNN references. It reads tracked results by default; reproducing training requires verified pretrained weights and the recorded parent checkpoint.

The [fresh local ResNet reproduction](results/resnet18-reproduction-v1.md) records verified frozen and layer4 runs, their learning curves, and comparison with the current SmallCNN references. Both checkpoints reproduce all saved validation predictions; test inference remains unused.

### ResNet optimisation and compute

The [bounded optimisation and compute record](results/resnet18-hour-v1.md) documents the selected learning rates, longer training, plateau scheduling, AdamW and late-stage adaptation. It compares validation macro-F1, per-class errors, missed faults and false alarms with SmallCNN, alongside recorded run times, phase timings, parameter counts and model-only latency. The [ResNet notebook](notebooks/03_resnet18_experiments.ipynb) contains the complete progression and executed plots. Test inference and additional-seed checks remain deferred.

The [generalisation experiment record](results/resnet18-generalisation-v1.md) compares mild augmentation, label smoothing, stronger rare-class weighting and gradual layer3 adaptation. The notebook records the hypotheses, fixed plans, matched continuation control, per-class results and both incremental and cumulative training costs. These remain validation-selected, single-seed experiments.
