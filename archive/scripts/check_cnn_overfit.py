"""Check whether SmallCNN can learn a tiny, fixed training subset."""

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from pv_fault.data import CLASS_NAMES, SolarModuleDataset
from pv_fault.models.baseline import SmallCNN
from pv_fault.train import evaluate_one_batch, train_one_batch, resolve_plan


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "configs/tiny-overfit-plan.json"
RUN_DIR = ROOT / "runs/smallcnn-tiny-overfit-s2112"


def main():
    plan = json.loads(resolve_plan(PLAN_PATH).read_text(encoding="utf-8"))
    if plan["data_split"] != "train" or plan["test_evaluated"] is not False:
        raise ValueError("This diagnostic must use training images only.")
    if RUN_DIR.exists():
        raise FileExistsError(f"Preserving existing result: {RUN_DIR}")

    torch.manual_seed(plan["seed"])
    dataset = SolarModuleDataset("train")

    selected_indices = []
    selected_ids = []
    for class_name in CLASS_NAMES:
        matches = [
            index for index, row in enumerate(dataset.rows)
            if row["label"] == class_name
        ]
        count = plan["images_per_class"]
        if len(matches) < count:
            raise ValueError(f"Too few training images for {class_name}")
        for index in matches[:count]:
            selected_indices.append(index)
            selected_ids.append(dataset.rows[index]["image_id"])

    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=len(selected_indices),
        shuffle=False,
    )
    images, labels = next(iter(loader))

    model = SmallCNN(num_classes=len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=plan["learning_rate"]
    )

    started = time.monotonic()
    history = []
    for step in range(1, plan["max_steps"] + 1):
        train_one_batch(model, images, labels, optimizer, criterion)

        if step % 20 == 0 or step == 1:
            result = evaluate_one_batch(model, images, labels, criterion)
            record = {
                "step": step,
                "loss": result["loss"],
                "accuracy": result["correct"] / result["count"],
                "count": result["count"],
            }
            history.append(record)
            print(json.dumps(record))

            if record["accuracy"] == 1.0 and record["loss"] < 0.05:
                break

        if time.monotonic() - started >= plan["max_seconds"]:
            break

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "result.json").write_text(
        json.dumps({
            "plan": plan,
            "class_names": CLASS_NAMES,
            "training_image_ids": selected_ids,
            "history": history,
            "elapsed_seconds": time.monotonic() - started,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved diagnostic to {RUN_DIR}")


if __name__ == "__main__":
    main()