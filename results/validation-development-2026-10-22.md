# Validation development — 22 October 2026

The submitted task is 12-class classification of InfraredSolarModules thermal
module crops using a small CNN and transfer-learned ResNet18. This record
documents development results only. The original test partition has not been
used in this cycle. Original split manifest SHA-256:
`8c6c80cc05943e85d4a03874b1d5d89cb83eaf98e5637560b9ec9a2dcdee1ef8`.
The split contains 13,946 train, 2,988 validation and 2,988 test images.
Known duplicate groups do not cross partitions. Panel/site/capture identities
are absent, so these results do not measure unseen-site generalization.

## Bounded learning-rate comparison

Three folds were made from the original training partition only, with
4,649/4,649/4,648 images. Each fold trained its own frozen ResNet18 head from
verified ImageNet weights before comparing layer-4 learning rates. Adam batch
64 and head/layer-4 rates 0.0001/0.00001 versus 0.0003/0.00003 were compared.
The second arm was informed by [Le et al. (2021)](https://www.sciencedirect.com/science/article/pii/S2213138821005579),
which used Adam at 0.0003 and batch 64 on the same dataset with a different
architecture. This is not a reproduction of that paper. The six fine-tuning
fits plus three fold-specific parent fits took 2,653.6 seconds on local CPU.

| Fold-mean validation metric | Current LR | Paper-informed LR |
|---|---:|---:|
| Macro-F1 | 0.5609 | 0.5808 |
| Accuracy | 0.7363 | 0.7556 |
| Balanced accuracy | 0.5669 | 0.5802 |
| Fault-to-normal rate | 0.1236 | 0.1227 |
| Normal false-alarm rate | 0.1070 | 0.0871 |

Macro-F1 improvements by fold were 0.0141, 0.0219 and 0.0237. The
fault-to-normal rates varied across folds and did not improve consistently.
This selected a single original-training-partition refit, without augmentation.
CV summary SHA-256:
`354f2623d3eb97effaf3506b95a19814fb065a9233930aad27d8592f8d852a7e`.

## Original validation after the selected refit

The selected refit trained for six epochs; best epoch 6, 463.3 seconds.
Its 2,988 saved validation predictions reproduce exactly from the saved
checkpoint. The selected model's validation accuracy was 0.7687, macro-F1
0.5965, balanced accuracy 0.5705, 206/1,492 faults classified No-Anomaly,
and 72/1,496 normal images classified as faults. Earlier unaugmented layer-4
ResNet18 validation accuracy was 0.7319, macro-F1 0.5629, balanced accuracy
0.5784, 184 faults classified No-Anomaly and 162 normal false alarms. Thus
the selected model improves macro-F1 but worsens missed faults under ordinary
12-class argmax. Small classes remain weak, including Cell-Multi recall 0.347,
Hot-Spot 0.486, and Soiling 0.387 on validation.

## Provisional technician-review routing

Keep the original 12-class argmax output. Separately, route an image as a
No-Anomaly **candidate for further testing** only if No-Anomaly is the argmax
and its softmax score is at least 0.7; otherwise route to qualified-technician
review. This 0.7 cutoff was selected *after inspecting validation curves* and
must be frozen before test. It is not a calibrated probability, risk limit,
safety clearance, or reuse decision.

| Validation routing | Earlier model, argmax | Selected model, 0.7 |
|---|---:|---:|
| Faults routed No-Anomaly | 184/1,492 | 119/1,492 |
| Normal images sent to review | 162/1,496 | 166/1,496 |
| All images sent to review | 1,470/2,988 | 1,539/2,988 |

At the selected 0.7 route, 119 of 1,449 No-Anomaly candidates are labeled
faulty in validation (8.2%). The route therefore cannot support any safety or
reuse claim. The independent pre-test reviewer conditionally accepted one
descriptive academic test cycle after model, routing, source hashes and test
access are locked. Expert fault-risk and review-capacity targets remain absent.

## External thermal check

The separate [Schaefer sUAS archive](https://zenodo.org/records/20860870)
contains 3,675 readable grayscale module crops with an incompatible fault
taxonomy. An exploratory binary projection of the earlier unaugmented
ResNet18 flagged 839/840 faults but also all 2,835 normals as faults after a
24×40 resize. This is evidence of severe domain shift, not a valid 12-class
or unseen-site test. The archive is from one orthomosaic. It was not used to
select the internal model or routing cutoff.

Next: lock code and checkpoint hashes, independently review the test evaluator
and test-access log, run one held-out test cycle, then prepare representative
Grad-CAM examples and measure inference on available hardware. No production,
electrical-safety, sustainability or unseen-site claim is supported yet.
