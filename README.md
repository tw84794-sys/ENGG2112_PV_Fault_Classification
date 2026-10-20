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
