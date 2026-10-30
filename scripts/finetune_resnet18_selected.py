"""Run the CV-selected plan through the shared ResNet fine-tuning command."""
from pathlib import Path
from finetune_resnet18 import main

if __name__ == "__main__":
    main(default_plan=Path("configs/experiment-plan-resnet18-selected.json"))
