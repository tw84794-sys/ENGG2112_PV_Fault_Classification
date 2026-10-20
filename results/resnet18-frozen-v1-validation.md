# Frozen ResNet18: validation comparison

A TorchVision ResNet18 with pinned ImageNet-1K V1 weights was adapted to the InfraredSolarModules 24 × 40 grayscale crops. Each image is bilinearly resized to 80 × 48, copied into three channels, then normalized with the weights' ImageNet mean and standard deviation. Resizing adds no thermal detail. The backbone and its BatchNorm statistics remain frozen; only the new 12-class linear head (6,156 parameters) is trained. This is a first transfer-learning comparison, not fine-tuning.

The run used the frozen duplicate-aware manifest (SHA-256 `8c6c80cc05943e85d4a03874b1d5d89cb83eaf98e5637560b9ec9a2dcdee1ef8`), seed 2112, train-only square-root inverse-frequency class weights, batch size 64, Adam learning rate 0.001, and unweighted validation macro-F1 for checkpoint selection. Five epochs ran on local CPU in 300.3 seconds; epoch 4 was selected. The plan allowed one trial, at most five epochs and 390 seconds.

| Validation measure | SmallCNN V3 | WideCNN V1 | Frozen ResNet18 V1 |
|---|---:|---:|---:|
| Accuracy | 0.5867 | 0.6583 | **0.6921** |
| Macro-F1 | 0.3318 | 0.3975 | **0.4900** |
| Balanced accuracy | 0.3690 | 0.3908 | **0.4925** |
| Faults predicted No-Anomaly | 336/1,492 (22.5%) | 458/1,492 (30.7%) | **299/1,492 (20.0%)** |
| Normal images predicted fault | 336/1,496 (22.5%) | **136/1,496 (9.1%)** | 156/1,496 (10.4%) |

ResNet18 improves macro-F1, accuracy and fault-to-normal errors against the wider CNN, while raising normal false alarms slightly. Some fault recall remains poor: Hot-Spot 6/37 (16.2%), Cell-Multi 49/193 (25.4%) and Soiling 10/31 (32.3%). These classes require error review before selection for any technician workflow.

After five warm-up passes, 20 model-only CPU inference measurements had a median of **3.00 ms/image** for batch size 1 and **162.94 ms/batch of 64** (about 393 images/s). This machine used four PyTorch CPU threads; timings exclude image loading, validation checks and application overhead. They are local measurements, not deployment capacity guarantees.

The pretrained weights came from [TorchVision's official ResNet18 URL](https://download.pytorch.org/models/resnet18-f37072fd.pth) and have SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. [TorchVision cautions](https://github.com/pytorch/vision#pre-trained-model-license) that pretrained models can carry dataset-derived terms; commercial-use rights have not been verified. The weights and full run files are ignored by Git. Fetch the exact weights with `scripts/fetch_resnet18_weights.py` before reproducing the run.

The checkpoint reproduced every one of the 2,988 saved validation predictions. **The test split was not evaluated.** This is a single-seed, duplicate-aware benchmark with no panel/site/capture identifiers, so it does not establish unseen-site or decommissioned-panel performance. Grad-CAM, operating thresholds, external validation and independent pre-test review remain to be done. No electrical safety, reuse, recycling or production-readiness claim follows from these results.
