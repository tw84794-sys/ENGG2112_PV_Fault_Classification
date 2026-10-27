# ResNet generalisation experiments

Five independent local CPU comparisons against the verified scheduled layer4 reference. The split, seed 2112, class order and test boundary are unchanged. Plans were fixed before full fits; each checkpoint was reloaded and reproduced all 2,988 validation predictions, including probabilities.

| Arm | Macro-F1 | Accuracy | Balanced accuracy | Missed faults | False alarms | Best/completed epochs | Fit min | Workflow min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.6261 | 79.08% | 59.66% | 196 | 55 | 10/10 | 15.47 | 20.44 |
| augmentation | 0.6281 | 78.65% | 60.46% | 198 | 52 | 10/10 | 12.30 | 17.26 |
| smoothing | 0.6170 | 77.68% | 61.20% | 137 | 114 | 7/10 | 14.57 | 19.54 |
| weight075 | 0.6259 | 78.61% | 59.48% | 195 | 67 | 10/10 | 13.26 | 18.23 |
| continue4 | 0.6267 | 78.82% | 60.63% | 151 | 93 | 1/4 | 5.31 | 25.75 |
| gradual34 | 0.6276 | 78.82% | 60.68% | 151 | 93 | 1/4 | 5.70 | 26.14 |

Selected by validation macro-F1: `resnet18-generalisation-augmentation-s2112-v1` (0.6281). The reference remains available even if a continuation performs worse.

Gradual layer3+4 minus matched layer4-only continuation: +0.0009 macro-F1. This paired comparison is the evidence for unfreezing another stage, rather than a comparison only with the older parent.

Actual new full-fit wall time across the five candidates: 51.15 minutes, excluding smoke checks, verification and reporting. Each fit includes initialization, image loading, training, two validation passes, metrics and saving. Workflow minutes additionally include the reused frozen-parent fit and, for continuations, the reference fit. ImageNet pretraining cost is excluded. Do not sum per-workflow costs to estimate search compute.

The input adapter and forward architecture are unchanged: 11,182,668 total parameters, with 8,394,764 trainable for layer4 or 10,491,916 for layer3+4. Prior architecture latency measurements still describe the compute scale; these new checkpoint weights were not benchmarked. No new claims about peak RAM, energy or FLOPs are made.

## Interpretation limits

- One seed only; repeated validation selection can overestimate generalisation.
- Ten-epoch fresh arms and four-epoch continuations have different histories; compare stage unfreezing with its matched continuation.
- Inherited training is reused across trials; per-workflow cumulative totals should not be summed as actual search cost.
- No peak RAM, energy or FLOPs measured. No new inference benchmark was run; all arms retain the same forward architecture.
- Recorded local CPU wall times vary with machine load; historical reference and new fits were not interleaved repeated timing trials.
- Augmented training accuracy and differently weighted/smoothed training losses are not directly comparable.
- Any selected validation improvement requires later replication; original test remains reserved.

## What we learned

Mild augmentation was the best new candidate: macro-F1 0.6281, a change of +0.0020 versus the reference. This is a small single-seed improvement. Accuracy fell to 78.65%; it missed 198 faults and produced 52 false alarms. Hot-Spot, Hot-Spot-Multi and Shadowing improved in F1, while Offline-Module and Vegetation worsened. Keep these limitations with the headline score.

Label smoothing 0.05 did not improve macro-F1 (0.6170), but changed the operating trade-off: 137 missed faults and 114 false alarms. It had the highest balanced accuracy among these candidates. It should not be described as uniformly worse on every metric.

The stronger class-weight exponent 0.75 was effectively tied in this run (0.6259), with more false alarms than the reference. This does not establish that class-weight adjustment never helps; only one alternative exponent was tested.

Gradual layer3+4 tuning improved macro-F1 by only +0.0009 relative to its matched layer4 continuation. Both selected their first additional epoch and had the same fault/normal error counts. More trainable parameters and the inherited training cost did not beat augmentation. These runs reset optimiser state; they are not exact resumes.

Retain the augmented candidate and the prior reference for later replication. Do not combine settings or treat this small validation gain as a confirmed test-set improvement. Additional seeds remain deferred, and original-test inference remains unused.

Parameter totals for the old reference were reconstructed from the shared model architecture because its historical producer did not record that field. Missing old phase timings are represented as null, not zero.
