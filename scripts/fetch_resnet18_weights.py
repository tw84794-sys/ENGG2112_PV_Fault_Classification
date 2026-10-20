"""Download and verify TorchVision's pinned ResNet18 ImageNet weights."""

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

from pv_fault.models.resnet import WEIGHT_SHA256, WEIGHT_URL


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/pretrained/resnet18-f37072fd.pth"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if DESTINATION.exists():
        digest = sha256_file(DESTINATION)
        if digest != WEIGHT_SHA256:
            raise ValueError("Cached ResNet18 weights fail SHA-256 verification.")
    elif args.verify_only:
        raise FileNotFoundError(DESTINATION)
    else:
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        temporary = DESTINATION.with_suffix(".pth.part")
        if temporary.exists():
            raise FileExistsError(
                f"Partial download already exists; inspect before retrying: {temporary}"
            )
        try:
            with urlopen(WEIGHT_URL, timeout=120) as source, temporary.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            digest = sha256_file(temporary)
            if digest != WEIGHT_SHA256:
                raise ValueError("Downloaded ResNet18 weights fail SHA-256 verification.")
            os.replace(temporary, DESTINATION)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Verified {DESTINATION} ({DESTINATION.stat().st_size} bytes)")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
