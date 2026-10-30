# Paired comparison of frozen models

The observed test macro-F1 difference, **ResNet18 − SmallCNN**, is **0.0275**.
Its 95% paired group-bootstrap interval is **0.0020 to 0.0524**.

The interval is above zero, supporting a positive macro-F1 gap on this benchmark.

The analysis reuses saved predictions for all 2,988 test images. Each of 1,000 resamples draws the same related-image groups for both models, preserving within-group rows and pairing. Settings match the original bootstrap: seed 2112, 95% percentile interval. No checkpoints or images are loaded and no inference, fitting or threshold selection occurs.

This analysis was added after observing the final test summaries. It conditions on the two frozen checkpoints and detected related-image groups. It cannot establish superiority across training seeds, independent sites, deployment hardware or operational objectives. A positive interval would still not meet the proposal's +0.05 improvement target when the observed gap is only +0.0275.

The [original evaluation](final-evaluation-v1.md) remains unchanged. Prediction and source fingerprints are retained in the [comparison record](final-model-comparison-v1.json); full saved predictions remain in the ignored local runs folder. Regeneration requires those prediction files, not model checkpoints.
