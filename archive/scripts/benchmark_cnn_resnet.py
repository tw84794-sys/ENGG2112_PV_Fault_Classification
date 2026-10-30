"""Compare model-only CPU latency on synthetic inputs after training finishes.

No dataset split is loaded. The output records checkpoint/source hashes and
raw timings; parameter storage is not peak RAM, FLOPs or energy consumption.
"""
import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pv_fault.models.baseline import SmallCNN
from pv_fault.models.resnet import ResNet18Thermal
from pv_fault.resnet_spatial import configure_spatial_variant


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def measure_batch(model, images):
    for _ in range(5):
        model(images)
    samples = []
    for _ in range(20):
        started = time.perf_counter_ns()
        model(images)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    median = statistics.median(samples)
    return {
        "batch_size": len(images), "median_batch_ms": median,
        "p90_batch_ms": float(np.percentile(samples, 90)),
        "amortized_median_ms_per_image": median / len(images),
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resnet_run", help="Verified ResNet run ID under runs/")
    args = parser.parse_args()
    resnet_run = (ROOT / "runs" / args.resnet_run).resolve()
    if not resnet_run.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Choose a ResNet run under runs/.")
    small_run = ROOT / "runs/smallcnn-depth-gentle60-s2112-v1"
    torch.set_num_threads(4)
    torch.manual_seed(2112)
    small = SmallCNN(num_classes=12, batch_norm=True, extra_conv=True, activation="relu")
    small.load_state_dict(torch.load(small_run / "best_model.pt", map_location="cpu",
                                    weights_only=True)["model_state_dict"])
    resnet = ResNet18Thermal(12, ROOT / "data/pretrained/resnet18-f37072fd.pth")
    plan = json.loads((resnet_run / "experiment-plan.json").read_text())
    checkpoint = torch.load(resnet_run / "best_model.pt", map_location="cpu", weights_only=True)
    variant = plan.get("spatial_variant", "standard")
    if checkpoint.get("spatial_variant", "standard") != variant:
        raise ValueError("Checkpoint spatial variant differs from its plan.")
    configure_spatial_variant(resnet, variant)
    resnet.load_state_dict(checkpoint["model_state_dict"])
    inputs = {size: torch.rand(size, 1, 40, 24) for size in (1, 64)}
    records = []
    for label, model, folder in (
        ("SmallCNN macro-F1 reference", small, small_run),
        ("Best phase ResNet", resnet, resnet_run),
    ):
        model.eval()
        record = {
            "model": label, "run_id": folder.name,
            "spatial_variant": variant if model is resnet else None,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "parameter_storage_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "checkpoint_sha256": file_hash(folder / "best_model.pt"),
            "batches": [measure_batch(model, inputs[size]) for size in (1, 64)],
        }
        records.append(record)
        print(label, record["total_parameters"], "parameters;",
              record["batches"][0]["median_batch_ms"], "ms/image;",
              record["batches"][1]["median_batch_ms"], "ms/batch64")
    result = {
        "measured_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_source_sha256": file_hash(Path(__file__)),
        "cpu_threads": 4, "device": "cpu", "torch": torch.__version__,
        "python": sys.version, "platform": platform.platform(), "test_evaluated": False,
        "scope": "Same synthetic grayscale inputs [N,1,40,24], model-only evaluation; no dataset images, "
                 "loading or application overhead. Five warm-ups then twenty measurements per batch size. "
                 "Run after all training.",
        "memory_scope": "Parameter storage only; excludes buffers, gradients, optimizer state and "
                        "activations. Not peak RAM, FLOPs or energy.",
        "models": records,
    }
    destination = ROOT / "results/resnet18-hour-v1-inference-compute.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
