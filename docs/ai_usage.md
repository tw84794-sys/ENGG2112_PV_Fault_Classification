# AI assistance record — working summary

Prepared 30 October 2026 from this project chat and the retained repository artifacts. Tool: OpenAI Codex. The exact underlying model/version for every earlier response has not been verified.

The assessment permits substantive AI assistance but requires prompts, outputs and the team's modifications or verification process in a report appendix. **This file is a summary, not a complete verbatim prompt/output archive or a completed compliance statement.** Preserve the original conversations and include the substantive records in the report appendix. Code outputs are retained in the referenced files and commits; conversational explanations and drafts must also be retained from the chat.

## Substantive assistance evidenced in this chat

Prompts below are quoted excerpts unless marked as contextual summaries. Outputs are summaries, not verbatim response transcripts.

| Work | User prompt or instruction | AI output / retained evidence | Automated verification performed by Codex |
|---|---|---|---|
| Repository simplification | “I want to minimise the amount of code by as much as possible. The jupyter notebooks are the most important part of the project.” | Refactored shared code, shortened notebooks and archived historical workflows. Commits `c0c8bdd` and `d17587b`. | Existing tests and notebook execution were checked during the changes; historical results and frozen split records were preserved. |
| Final model and screening policy | “Standard augmented ResNet (recommended)” and “Select screening threshold on validation (recommended)” | [Frozen protocol](../configs/final-evaluation-v1.json), [evaluation script](../scripts/evaluate_final.py) and [final results](../results/final-evaluation-v1.md). Freeze commit `3ef4040`; result commit `9fa1067`. | Reproduced saved validation probabilities; committed policy before test inference; verified saved test identities, labels, metrics and hashes. The then-current 46 tests and three notebooks passed. |
| Grad-CAM | “alrigth should we push what we have now and then have a go at gradcam” | [Grad-CAM helper](../src/pv_fault/gradcam.py), notebook 03 section 9 and commit `116aaab`. | Analytic toy-model tests, fixed-weight and hook-cleanup checks, reproduction of selected validation predictions within measured floating-point tolerance, executed notebook and visual inspection. The then-current 49 tests passed. |
| Crop robustness | Contextual summary: user accepted the proposed validation-only shift, rotation, blur and resolution-loss check with “alright lets do that”. | [Robustness script](../scripts/check_validation_robustness.py), [record](../results/resnet18-validation-robustness-v1.json), notebook 03 section 10 and commit `2a03d61`. | Unchanged validation predictions matched exactly. Nine conditions retained identical image identities, labels and groups; saved metrics and hashes were recomputed. Transform tests and notebook execution with/without local data were checked. The then-current 53 tests passed. |
| Report planning and limitations | Contextual summary: user supplied the Findings and Recommendations rubric and asked how to address it, then discussed small thermal crops and deployment preprocessing. | Chat guidance connected objectives to results, uncertainty, error analysis, compute costs, stakeholders and limitations. No completed report was produced. A later [Markdown scaffold](report_outline.md), requested by the user from the supplied LaTeX outline and rubric, adds headings and topic bullets only. | Repository results and primary sources were consulted. Submitted proposal targets, physical fault labels, real-camera compatibility and environmental/economic benefits were not independently established. |
| Paired comparison and documentation | “if you agree and think its worthwile start making the additions locally and ill review them before commiting etc” | [Paired analysis](../scripts/compare_frozen_models.py), [comparison record](../results/final-model-comparison-v1.json), notebook subsection, README summary, dependency pins and author identity mapping. Initially prepared as uncommitted changes for user review; subsequent commit/push authorised in the project chat. | Same-group paired resampling; identity, metric and fingerprint checks against frozen records; tests cover zero gaps, direction, group integrity and invalid pairing. No new test inference or checkpoint loading. |

## Human decisions and review still required

The user explicitly selected the standard augmented ResNet and validation-selected screening policy. The user also requested the Grad-CAM and robustness additions and reviewed examples and progress in the chat. Those interactions do not establish a complete independent review of every generated line or interpretation.

Before submission, the team should:

- Retain the original substantive prompts and outputs, including generated code/diffs, explanations and report wording that is used. Add any other AI tools or conversations not represented here.
- Record its actual modifications, independent checks, accepted/rejected suggestions and responsibilities. Do not substitute automated checks for human verification.
- Verify targets against the submitted proposal, check citations against original sources, and ensure that claims distinguish validation, final test and post-hoc analysis.
- Confirm all team members and contributions, including work not reflected in Git; the available proposal draft lists Iris Li, Ethan Melton, Anushka Ram and Tom Wait. Commit author names are not proof that code was written without AI assistance.
- Prepare the required report appendix and retain the supporting material as directed by the assessment brief.

## Limits on what was verified

Tests and reproduced predictions check software behaviour and recorded measurements. They do not establish electrical safety, correct physical fault labels, unseen-site performance, calibrated confidence, real camera/cropping tolerances, or quantified financial/environmental benefits. Detected related-image groups are not verified independent sites. The paired comparison conditions on the two trained checkpoints and was added after observing the final test summaries.
