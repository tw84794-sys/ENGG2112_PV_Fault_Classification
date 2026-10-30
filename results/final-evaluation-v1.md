# Final reserved-test evaluation

All models, screening thresholds and bootstrap settings were [frozen](../configs/final-evaluation-v1.json) before test inference. These results do not select new settings.

| Model | Accuracy | Macro-F1 | Balanced accuracy | Argmax missed faults | Argmax normal false alarms | Screening recall | Screening false-alarm rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority baseline | 50.07% | 0.0556 | 8.33% | 1492 | 0 | 0.00% | 0.00% |
| SmallCNN | 76.51% | 0.5957 | 58.54% | 193 | 85 | 89.68% | 7.62% |
| ResNet18 | 78.85% | 0.6232 | 60.18% | 210 | 43 | 89.48% | 4.75% |

Multiclass predictions use argmax. Screening is a separate binary decision using thresholds selected solely on validation. The majority baseline predicts the most frequent training class and has no fitted threshold.

## Frozen screening decisions

| Model | Threshold | Missed faults | Normal false alarms | Recall | False-alarm rate | Both test targets met |
|---|---:|---:|---:|---:|---:|---|
| Majority baseline | constant class | 1492 | 0 | 0.00% | 0.00% | False |
| SmallCNN | 0.555954 | 154 | 114 | 89.68% | 7.62% | False |
| ResNet18 | 0.392924 | 157 | 71 | 89.48% | 4.75% | False |

Targets are at least 90% anomaly recall and at most 20% normal false alarms. They are research goals from the proposal, not safety standards. No threshold is changed after inspecting test results.

## Uncertainty and limitations

| Model | Macro-F1 95% interval | Screening recall 95% interval | False-alarm rate 95% interval |
|---|---|---|---|
| Majority baseline | 0.0538–0.0574 | 0.0000–0.0000 | 0.0000–0.0000 |
| SmallCNN | 0.5660–0.6237 | 0.8814–0.9113 | 0.0616–0.0916 |
| ResNet18 | 0.5939–0.6515 | 0.8785–0.9084 | 0.0362–0.0591 |

The JSON retains per-class precision, recall, F1, supports, confusion matrices and 95% group-bootstrap intervals. Groups reflect detected visual relationships, not independently sampled sites. These intervals do not establish unseen-site generalisation. The models were selected through repeated validation experiments, mainly with one seed. Predictions cannot certify panel safety or reuse suitability.

![Reserved-test confusion matrices](final-evaluation-v1-confusion.png)

Frozen protocol SHA-256: `b0e1c1fdee9be29fb79919fbf23bc8b15da18833f12dd3ecadcbf398786ce21d`. Evaluation code commit: `3ef4040e232ff8e5945c5989153f48401a8a5949`.

Full image-level predictions remain in the ignored `runs/final-evaluation-v1/` folder, with their hashes in the JSON. The original validation records remain unchanged.
