"""Small displays shared by the experiment notebooks."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pv_fault.data import CLASS_NAMES


def learning_curves(runs, *, title, key="validation_macro_f1", labels=None, ax=None):
    own_axes = ax is None
    if own_axes:
        _, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for index, run in enumerate(runs):
        history = run["history"]
        label = labels[index] if labels is not None else run.get("arm", run["run_id"])
        ax.plot([row["epoch"] for row in history], [row[key] for row in history], label=label)
    ax.set(title=title, xlabel="Epoch", ylabel=key.replace("_", " "))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    if own_axes:
        plt.show()
    return ax


def confusion_matrix(metrics, title, ax=None):
    own_axes = ax is None
    if own_axes:
        _, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    counts = np.asarray(metrics["confusion_matrix"])
    percentages = np.divide(100 * counts, counts.sum(axis=1, keepdims=True),
                            out=np.zeros_like(counts, dtype=float),
                            where=counts.sum(axis=1, keepdims=True) != 0)
    shown = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    ax.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted class", ylabel="True class", title=title)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.figure.colorbar(shown, ax=ax, label="Percentage of true class")
    if own_axes:
        plt.show()
    return ax


def image_examples(examples, image_root, title="Validation examples"):
    if not examples:
        return
    if not all((Path(image_root) / row["image_path"]).is_file() for row in examples):
        print("Image examples skipped: prepare the local dataset using data/README.md.")
        return
    _, axes = plt.subplots(1, len(examples), figsize=(2.5 * len(examples), 4.5),
                           layout="constrained", squeeze=False)
    for ax, row in zip(axes.flat, examples):
        with Image.open(Path(image_root) / row["image_path"]) as image:
            ax.imshow(np.asarray(image) / 255, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"ID {row['image_id']}\nTrue: {row['true_label']}\nPred: {row['predicted_label']}", fontsize=9)
        ax.axis("off")
    axes.flat[0].figure.suptitle(title)
    plt.show()
