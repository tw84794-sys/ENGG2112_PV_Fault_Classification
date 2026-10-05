# Contributing

Coordinate file ownership before editing. Make focused changes on a branch from the current default branch, keep commits reviewable and describe what changed, why and which checks ran. Do not overwrite another teammate's work.

Keep model code in `src/pv_fault/`, command entry points in `scripts/`, settings in `configs/`, and meaningful invariant checks in `tests/`. Notebooks should call reusable code rather than becoming a second training implementation. Update dataset/evaluation records when assumptions change.

Raw images, environments, credentials, checkpoints and full run outputs must stay outside Git. Review the frozen split, training/validation boundary and saved metrics before accepting model changes. Do not claim a check passed when it was skipped. Agree any dependency lock, CI job or hardware-specific setup as a team before making it mandatory.
