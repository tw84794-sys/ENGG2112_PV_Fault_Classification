# Wider CNN: validation comparison

Both runs used the frozen InfraredSolarModules manifest, seed 2112, train-only square-root inverse-frequency class weights, batch size 64, learning rate 0.001, and validation macro-F1 checkpoint selection. The wider model changes only layer widths (convolution channels 16/32 to 64/128; classifier hidden width 64 to 128).

| Validation measure | SmallCNN V3 | WideCNN V1 |
|---|---:|---:|
| Trainable parameters | 38,412 | 338,316 |
| Selected epoch | 13 | 14 |
| Accuracy | 0.5867 | 0.6583 |
| Macro-F1 | 0.3318 | 0.3975 |
| Balanced accuracy | 0.3690 | 0.3908 |
| Faults predicted No-Anomaly | 336/1,492 (22.5%) | 458/1,492 (30.7%) |
| Normal images predicted fault | 336/1,496 (22.5%) | 136/1,496 (9.1%) |

The wider CNN improved validation accuracy and macro-F1, but missed more faults as No-Anomaly. Rare-class recall remains weak: Hot-Spot 1/37, Diode-Multi 2/26, and Soiling 8/31. The selected checkpoint therefore does not satisfy the full hypothesis or justify technician use without review.

The V4 trial completed 14 of 15 planned epochs before the 120-second cap; total elapsed time was 126.2 seconds because the cap is checked after each epoch. Both runs are single-seed comparisons. The test split was not evaluated. No panel/site/capture identifiers are available, so this duplicate-aware validation benchmark does not establish unseen-site performance. Full local run files, including predictions and the checkpoint, are under `runs/widecnn-s2112-cpu-v1-weighted/` and excluded from Git.
