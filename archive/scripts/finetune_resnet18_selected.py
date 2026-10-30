"""Run the CV-selected plan through the shared ResNet fine-tuning command."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from finetune_resnet18 import main

if __name__ == "__main__":
    main(default_plan=Path("configs/experiment-plan-resnet18-selected.json"))
