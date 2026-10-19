"""Loader contracts using tiny local images, without downloading the dataset."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_fault.data import CLASS_NAMES, SolarModuleDataset, make_data_loaders


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        self.manifest = self.root / "split.csv"
        self.rows = []
        metadata = {}
        for split in ("train", "val", "test"):
            for label in CLASS_NAMES:
                image_id = str(len(self.rows))
                path = f"images/{image_id}.png"
                Image.new("L", (24, 40), 128).save(self.root / path)
                metadata[image_id] = {"image_filepath": path, "anomaly_class": label}
                self.rows.append(dict(image_id=image_id, image_path=path,
                                      label=label, group_id=image_id, split=split))
        (self.root / "module_metadata.json").write_text(json.dumps(metadata))
        self.write_manifest()

    def write_manifest(self):
        with self.manifest.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        summary = {"images": len(self.rows)}
        for path, key in ((self.manifest, "manifest_sha256"),
                          (self.root / "module_metadata.json", "metadata_sha256")):
            summary[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest.with_suffix(".json").write_text(json.dumps(summary))

    def dataset(self, split="train", **kwargs):
        return SolarModuleDataset(split, data_root=self.root,
                                  manifest=self.manifest, **kwargs)

    def test_grayscale_scaling_and_shared_class_order(self):
        for split in ("train", "val", "test"):
            dataset = self.dataset(split)
            image, label = dataset[0]
            self.assertEqual(image.shape, (1, 40, 24))
            self.assertEqual(image.dtype, torch.float32)
            self.assertTrue(torch.allclose(image, torch.full_like(image, 128 / 255)))
            self.assertEqual(label, 0)
            self.assertEqual([dataset[i][1] for i in range(12)], list(range(12)))

    def test_rejects_cross_split_group_even_with_valid_checksum(self):
        self.rows[12]["group_id"] = self.rows[0]["group_id"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            self.dataset()

    def test_rejects_changed_manifest(self):
        with self.manifest.open("a") as target:
            target.write("\n")
        with self.assertRaisesRegex(ValueError, "Fingerprint mismatch"):
            self.dataset()

    def test_rejects_rgb_instead_of_silently_converting(self):
        Image.new("RGB", (24, 40)).save(self.root / self.rows[0]["image_path"])
        with self.assertRaisesRegex(ValueError, "Expected grayscale"):
            self.dataset()[0]

    def test_augmentation_is_train_only(self):
        for split in ("val", "test"):
            with self.assertRaisesRegex(ValueError, "only allowed"):
                self.dataset(split, transform=lambda image: image)
        self.assertTrue(torch.equal(self.dataset(transform=lambda x: x * 0)[0][0],
                                    torch.zeros(1, 40, 24)))

    def test_batches_shuffle_training_only_and_keep_all_examples(self):
        loaders = make_data_loaders(batch_size=5, data_root=self.root,
                                    manifest=self.manifest)
        self.assertIsInstance(loaders["train"].sampler, RandomSampler)
        for split in ("val", "test"):
            self.assertIsInstance(loaders[split].sampler, SequentialSampler)
        for loader in loaders.values():
            batches = list(loader)
            self.assertEqual([len(labels) for _, labels in batches], [5, 5, 2])
            self.assertEqual(batches[0][1].dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
