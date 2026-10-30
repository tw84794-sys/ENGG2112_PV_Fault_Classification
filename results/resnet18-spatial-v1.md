# ResNet early-downsampling experiments

Same weights, frozen-parent head, split, seed, augmentation, optimiser, learning rates, schedule, batch size and frozen BatchNorm. Change exactly one early-downsampling operation.

| Variant | Macro-F1 | Accuracy | Balanced accuracy | Missed faults / 1492 | False alarms / 1496 | Best/completed epochs | Fit min | Median batch-one ms | Median batch-64 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard | 0.6281 | 78.65% | 60.46% | 198 | 52 | 10/10 | 12.30 | 2.610 | 224.018 |
| stride1 | 0.6303 | 78.95% | 60.40% | 205 | 51 | 10/10 | 21.20 | 4.383 | 357.416 |
| no_maxpool | 0.6229 | 78.25% | 61.45% | 159 | 91 | 10/10 | 23.00 | 4.203 | 326.831 |

Selected by validation macro-F1: `resnet18-spatial-stride1-s2112-v1`. The standard augmented reference remains eligible; a larger grid is not automatically adopted.

Stride1 retains the 7×7 kernel tensors and changes convolution stride to one. No_maxpool retains convolution stride two and replaces only the initial max pool with Identity. Both preserve a 512×5×3 layer4 grid versus 512×3×2 for standard ResNet. Global average pooling and the classifier are unchanged. Total and trainable parameter counts are unchanged; activations and operations increase.

The two new full fits used 44.21 local CPU wall minutes in total, excluding smoke checks, external checkpoint verification, benchmarks and reporting. Each includes initialization, training with loading, two validation passes, metrics and saving. A fresh workflow also includes the reused frozen-head parent (4.97 recorded minutes); ImageNet pretraining cost is excluded. All runs used four CPU threads. Local load varies; historical fit times were not interleaved repeated timing trials.

Actual selected checkpoints were benchmarked after all training on identical synthetic [N,1,40,24] inputs, with five warmups and twenty measured forwards per batch size. No dataset images were loaded for the benchmark. Raw timings, source/checkpoint hashes, feature geometries and SmallCNN measurements are retained separately. Parameter storage is not peak RAM; no FLOPs or energy were measured.

## Interpretation and remaining work

This tests applying each architectural change during layer4 adaptation while earlier parameters and BatchNorm statistics remain frozen. The shared head was fitted with the standard stem, which can initially disadvantage modified stems; this does not exhaust retraining with stem-specific head/BN adaptation.

Each architecture was applied before any updates in its independent run, starting from the same frozen-head checkpoint as the augmented reference. The reference is not a warm-start parent for these comparisons. Larger early grids change activation distributions even with identical pretrained tensors, and fixed BatchNorm statistics may become less suitable. A result here cannot establish whether the variant would help with stem-specific head refitting or broader adaptation.

Full per-class precision/recall/F1, confusion matrices and epoch histories are retained in the JSON and notebook. Checkpoints reproduced all 2,988 validation predictions including probabilities. Variant metadata is explicitly restored during verification because stride and pooling are absent from learned state dictionaries. One seed only; repeated validation selection can overestimate gains. Additional seeds remain deferred, and original-test inference remains unused.

## Decision after both fits

Stride1 increased validation macro-F1 by +0.0022, from 0.6281 to 0.6303. Accuracy changed from 78.65% to 78.95%, while missed faults / false alarms changed from 198 / 52 to 205 / 51. Balanced accuracy was essentially unchanged. This is a small single-seed gain, with worse missed-fault performance, not an improvement on every metric.

Stride1 took 21.20 fit minutes versus 12.30 recorded reference minutes. Its measured batch-one latency was 1.68 times the standard model, and batch-64 latency 1.60 times. Parameter count stayed fixed; preserving more spatial activations increased actual compute.

Removing max pooling did not beat the reference on macro-F1 (0.6229), although it reduced missed faults to 159 and increased balanced accuracy to 61.45%. False alarms rose to 91. It therefore represents another error trade-off, not a uniform decline on every metric.

Keep the standard augmented reference as the lower-cost option and stride1 as the highest-macro-F1 candidate for later replication. Do not automatically replace the model based on this small validation gain. The deterministic weak-class review and saved example images in the notebook should guide any next hypothesis. A spatial-grid classifier, stem-specific head refit, BatchNorm adaptation and additional seeds were not run in this batch.
