# Baseline CNN error review and handoff

Original validation only; descriptive review after model selection, no new training or test inference.

Retain two three-layer BatchNorm/ReLU SmallCNN references. Neither dominates all measures.

| Role | Macro-F1 | Accuracy | Missed faults / 1,492 | Normal false alarms / 1,496 |
|---|---:|---:|---:|---:|
| Fewer missed faults | 0.5964 | 0.7413 | 82 | 240 |
| Highest macro-F1 | 0.6012 | 0.7634 | 170 | 112 |

## Fewer missed faults

Run: `smallcnn-search-bn-depth-s2112-v1`.

Weakest classes by F1 (support is the number of validation examples):

| Class | F1 | Recall | Support |
|---|---:|---:|---:|
| Soiling | 0.313 | 0.419 | 31 |
| Hot-Spot | 0.435 | 0.405 | 37 |
| Cell-Multi | 0.440 | 0.409 | 193 |
| Offline-Module | 0.484 | 0.634 | 123 |

Most frequent directional confusions (counts, influenced by class size):

- No-Anomaly → Shadowing: 93
- No-Anomaly → Offline-Module: 88
- Cell-Multi → Cell: 38
- Vegetation → Cell: 35
- Cell → Vegetation: 32

## Highest macro-F1

Run: `smallcnn-depth-gentle60-s2112-v1`.

Weakest classes by F1 (support is the number of validation examples):

| Class | F1 | Recall | Support |
|---|---:|---:|---:|
| Soiling | 0.355 | 0.355 | 31 |
| Cell-Multi | 0.429 | 0.383 | 193 |
| Hot-Spot | 0.434 | 0.486 | 37 |
| Offline-Module | 0.516 | 0.472 | 123 |

Most frequent directional confusions (counts, influenced by class size):

- Cell-Multi → Cell: 52
- Offline-Module → No-Anomaly: 49
- Vegetation → Cell: 44
- Shadowing → No-Anomaly: 38
- No-Anomaly → Offline-Module: 31

## Handoff decisions

- Keep the 0.5964 reference when comparing missed-fault counts; keep the 0.6012 scheduler model as the macro-F1 reference. These are development references, not deployment selections.
- A wrong fault type remains wrong for the 12-class task even when the model correctly recognises an anomaly.
- The notebook shows deterministic illustrative errors (lowest image ID in each error category), not a random or representative error-rate sample. A visual example does not establish the cause of an error or prove a label is incorrect.
- Small validation supports make rare-class estimates uncertain. Repeated selection on this validation set also makes gains optimistic. Additional seed checks remain deferred.
- Continue with ResNet in its separate notebook using the same frozen split and metric definitions. Keep test inference reserved until model choices are frozen.
- Checkpoints and full predictions remain local in ignored runs folders. The tracked review and notebook preserve the findings without requiring those files.
