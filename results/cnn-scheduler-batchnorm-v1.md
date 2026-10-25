# Scheduler and BatchNorm CNN comparison — 23 October 2026

Two separate changes were tested for both SmallCNN and WideCNN: a validation-macro-F1 plateau scheduler, or BatchNorm after each convolution. Controls are the immediately preceding constant-0.001 runs in the same Python/PyTorch environment. Tests confirm identical initial convolution/classifier weights for the BatchNorm variants, while default models retain their original parameter counts and architecture.

Each new trial used Adam at initial LR 0.001, batch 64, seed 2112, four CPU threads, training-only square-root inverse-frequency class weights, no augmentation, a 30-epoch maximum, early-stopping patience five, and 300 seconds checked between epochs. The scheduler uses mode=max, patience=1, factor=0.3, threshold=0 and min_lr=0.00001. BatchNorm trials keep a constant learning rate. Biases are retained in convolutions to isolate the addition of BatchNorm. No SGD/GELU trials or combined scheduler+BatchNorm trials were run.

| Model | Arm | Accuracy | Macro-F1 | Delta F1 vs control | Faults called normal / 1,492 | Normal false alarms / 1,496 | Epochs / best | Stop |
|---|---|---:|---:|---:|---:|---:|---:|---|
| small_cnn | control | 0.6412 | 0.4224 | +0.0000 | 343 | 245 | 30 / 29 | max_epochs |
| small_cnn | scheduler | 0.6222 | 0.3667 | -0.0556 | 404 | 219 | 30 / 28 | max_epochs |
| small_cnn | batchnorm | 0.6523 | 0.4888 | +0.0664 | 152 | 349 | 18 / 13 | early_stopping |
| wide_cnn | control | 0.6914 | 0.5114 | +0.0000 | 258 | 199 | 30 / 27 | max_epochs |
| wide_cnn | scheduler | 0.7011 | 0.5209 | +0.0095 | 287 | 181 | 30 / 27 | max_epochs |
| wide_cnn | batchnorm | 0.6727 | 0.5051 | -0.0063 | 129 | 334 | 18 / 13 | early_stopping |

## Interpretation

- **small_cnn / scheduler:** macro-F1 changed by -0.0556; missed faults changed by +61; normal false alarms changed by -26. LR reductions recorded: 4; epochs actually trained below the initial LR: 15. Selected training accuracy 0.6267, validation accuracy 0.6222.
- **small_cnn / batchnorm:** macro-F1 changed by +0.0664; missed faults changed by -191; normal false alarms changed by +104. LR reductions recorded: 0; epochs actually trained below the initial LR: 0. Selected training accuracy 0.6740, validation accuracy 0.6523.
- **wide_cnn / scheduler:** macro-F1 changed by +0.0095; missed faults changed by +29; normal false alarms changed by -18. LR reductions recorded: 2; epochs actually trained below the initial LR: 4. Selected training accuracy 0.7257, validation accuracy 0.7011.
- **wide_cnn / batchnorm:** macro-F1 changed by -0.0063; missed faults changed by -129; normal false alarms changed by +135. LR reductions recorded: 0; epochs actually trained below the initial LR: 0. Selected training accuracy 0.7015, validation accuracy 0.6727.

A scheduler decision at the end of an epoch affects the next epoch, not the one just completed. A reduction on the final epoch cannot improve that completed run. Retain a change only with its error trade-offs documented; higher accuracy alone is insufficient. These results support configurations within this budget and seed, not universal optimiser or architecture rankings.

## Checks and reproducibility

- All four selected checkpoints were reloaded and exactly reproduced every saved validation prediction, including probabilities. BatchNorm checkpoint metadata records `batch_norm`; reconstruct the matching model with that flag when loading.
- Split/image-byte verification passed before training. No test image inference or test-based model selection was performed.
- The suite ran 21 tests: 19 passed, with two existing skips (the old checkpoint placeholder and a ResNet test requiring absent pretrained weights). New tests cover preserved initial weights, BatchNorm validation/checkpoint behaviour, and scheduler direction/timing.
- Current default models preserve prior control behaviour, but source fingerprints differ from controls because optional features were added. Per-run hashes are retained; the older report’s source-match statement applies at its original recording time.
- Checkpoint and image-level prediction files remain in ignored runs folders. The tracked JSON provides histories, metrics, plans and provenance. Tests confirm validation does not update BatchNorm running statistics.
- Original validation has been reused for development. Additional seeds/training-only folds should confirm promising changes. No unseen-panel/site or safety claim is established.

Reproduce one planned trial after preparing the dataset (replace PLAN with an actual scheduler or batchnorm plan):

```bash
PYTHONPATH=src python scripts/train_baseline.py --plan PLAN
PYTHONPATH=src python scripts/train_baseline.py --plan PLAN --full
```

Full runs refuse to overwrite existing folders. See `notebooks/02_baseline_experiments.ipynb` for the chronological decision log.
