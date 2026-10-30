# ENGG2112 final report scaffold

## Writing plan — guidance outside the submitted report

### Structure and page budget

- Supplied LaTeX section order; project-specific subheadings; short topic prompts
- Maximum 10 body pages; title, executive summary, contents, references and appendices excluded
- Suggested body allocation: sections 1–3, 2 pages; section 4, 2.5 pages; section 5, 3 pages; section 6, 2 pages; section 7, 0.5 pages
- Core findings, explanations and stakeholder recommendations in the body; supporting detail in appendices

### Assessment coverage and evidence

- Objectives, motivation and prior work: sections 1–3; 4 marks
- Methods, technical justification and adaptations from the proposal: section 4; 4 marks
- Findings, objective-to-result links, errors and limitations: section 5; 4 marks
- Recommendations, stakeholders, deployment, impact and ethics: section 6; 3 marks
- Submitted-proposal confirmation; draft targets and planned responsibilities distinguished from verified commitments and actual contributions

## Title Page

### Project and team details

- Project Report for ENGG2112; concise project title within two lines
- Proposed title: Thermal-image classification of photovoltaic module anomalies for technician-led screening
- Confirmed team names, student IDs and disciplines; university/faculty details; submission date
- Public repository link: [ENGG2112 PV Fault Classification](https://github.com/tw84794-sys/ENGG2112_PV_Fault_Classification)

## Executive Summary

### Problem, approach, outcomes and recommendation

- Technician-led thermal anomaly screening; inspection and prospective reuse-assessment context
- Small, imbalanced module crops; majority baseline → SmallCNN → pretrained ResNet18
- Final test macro-F1: CNN 0.5957, ResNet 0.6232; screening recall 89.48%, false alarms 4.75% for ResNet
- Research targets missed; positive paired benchmark gap; substantial compute trade-off
- Human-reviewed triage; independent-site and complete-pipeline evaluation before deployment

## Table of Contents

### Navigation

- Numbered template sections and project-specific subsections
- Figure/table captions; appendix entries; final page numbering

## 1. Background and Motivation

### 1.1 PV inspection and the practical problem

- PV module anomalies; thermal patterns; inspection workload and prioritisation
- Reuse, maintenance and waste-management motivation; evidence-backed context
- Thermal anomaly labels versus permanent damage, electrical safety and reuse suitability

### 1.2 Intended users and project scope

- Inspection technicians; operators; prospective refurbishers and recyclers
- Classification of already-cropped thermal module images
- Inspection prioritisation as intended use; qualified physical/electrical assessment as a separate stage
- Offline prototype boundaries; potential benefits versus demonstrated outcomes

## 2. Objectives and Problem Statement

### 2.1 Prediction tasks and research questions

- Input: one grayscale thermal module crop, 24 pixels wide × 40 pixels high
- Output: one of 12 thermal-condition classes, including No-Anomaly
- Binary screening score: 1 − P(No-Anomaly); anomaly-positive decision above a fixed cutoff
- Transfer-learning benefit versus SmallCNN; missed-fault, false-alarm and compute trade-offs

### 2.2 Success criteria and deliverables

- Proposal-draft targets: macro-F1 ≥ 0.70; improvement over CNN ≥ 0.05
- Screening targets: anomaly recall ≥ 90%; normal-image false-alarm rate ≤ 20%
- Reproducible cleaning/split workflow; baselines; frozen final evaluation; demonstration notebooks
- Error analysis, uncertainty estimates, Grad-CAM and limitations
- Research targets versus safety requirements; confirmation against the submitted proposal

## 3. Related Work

### 3.1 Thermal inspection and the source dataset

- Raptor Maps InfraredSolarModules repository and dataset paper
- Aerial thermal inspection; individual-module cropping; low-resolution and class-imbalance constraints
- Published full inspection pipelines; detection, segmentation and site-level evaluation
- Differences in datasets, splits and tasks; limits on direct score comparisons

### 3.2 Model and explanation choices

- Compact CNNs as low-cost learned baselines
- ImageNet-pretrained ResNet18; transfer from visible-spectrum features to thermal crops
- Grad-CAM as class-specific qualitative evidence; coarse localisation and interpretation limits
- Project contribution: controlled comparisons, frozen test outcomes and explicit operational trade-offs

## 4. Methodology

### 4.1 Data Pre-Processing

- Source, licence and provenance; 20,000 original images across 12 classes
- Corrupt-image/label checks; exact-duplicate removal; conflicting-label and near-duplicate handling
- Retained dataset: 19,922 images; exclusions by reason from notebook 01
- Frozen split: 13,946 training / 2,988 validation / 2,988 test images; stratification and related-image grouping
- Missing verified panel/site/capture identities; unresolved independence limitations

### 4.2 Feature Extraction and Image Representation

- Native grayscale tensors; brightness scaling from 0–255 to 0–1
- CNN spatial features learned from training images
- ResNet adapter: 40 × 24 → 80 × 48; repeated grayscale channels; ImageNet normalisation
- Upsampling without added thermal detail; standard ResNet final feature grid of 3 × 2

### 4.3 Classification and Model Development

- Training-majority baseline; rationale under class imbalance
- SmallCNN architecture and retained macro-F1 reference
- ResNet18: verified pretrained weights; replaced 12-class head; frozen-backbone phase → layer4 adaptation
- Training-only class weighting; frozen BatchNorm during late-stage adaptation
- Controlled development comparisons: scheduling, regularisation, augmentation and early spatial downsampling

### 4.4 Training, Validation and Screening Policy

- Validation-based checkpoint/model selection; learning-rate schedule and stopping rules
- Training-only augmentation; shared split and relevant matched controls
- Standard augmented ResNet selection; validation performance versus compute and missed-fault trade-offs
- Highest validation cutoff meeting 90% anomaly recall; resulting false alarms checked against 20%
- Frozen choices and code committed before final test inference; unchanged policy afterwards

### 4.5 Simulation Environment and Evaluation

- Local CPU environment; four threads; recorded Python/dependency versions
- Accuracy, macro-F1, balanced accuracy, class precision/recall/F1 and support counts
- Row-normalised confusion matrices; argmax outcomes versus separate binary screening outcomes
- Checkpoint/source/data fingerprints; exact validation-probability reproduction
- Measured inference/fit timing scope; hardware and batch size; exclusions from cost estimates

### 4.6 Uncertainty and Additional Analyses

- 95% group-bootstrap intervals; 1,000 resamples; seed 2112; detected related-image groups
- Post-hoc paired macro-F1 gap; identical sampled groups for both frozen models; saved test predictions only
- Deterministic validation Grad-CAM examples; predicted-class and true-class maps
- Validation crop sensitivity: four 2-pixel shifts, ±5° rotations, blur and half resolution; one disturbance per condition
- Fixed robustness checkpoint, threshold and labels; exact unchanged-image reference check

### 4.7 Adaptations from the Proposed Work Plan

- Planned versus completed milestones; retained, modified and deferred elements
- CPU-compatible workflow; bounded development trials; evidence-based model choice
- Visual grouping in place of unavailable verified site/panel grouping
- Additional crop-sensitivity and post-hoc paired analyses; distinct purposes and timing
- Completed notebook demonstration versus remaining operational-system components

## 5. Simulation Results

### 5.1 Key Findings and Objective Attainment

- Main table: majority baseline, SmallCNN and standard ResNet; final test metrics and key intervals
- ResNet test accuracy 78.85%, macro-F1 0.6232; CNN accuracy 76.51%, macro-F1 0.5957
- Observed macro-F1 gain +0.0275 versus draft +0.05 target; ResNet score below 0.70 target
- Screening recall 89.48% versus 90% target; false alarms 4.75% versus 20% limit
- Objective → measured outcome → interpretation table; no post-test retuning

### 5.2 Development Findings and Compute Trade-Offs

- Selected validation progression: majority/CNN → frozen ResNet → late-stage adaptation
- Standard versus stride 1: validation macro-F1 0.6281 versus 0.6303; limited single-seed gain
- Stride 1: increased measured inference time; more argmax missed faults than standard
- ResNet versus CNN: approximately 234× parameters; 7.3× local batch-one latency
- Validation ablations versus final test evidence; benchmark hardware/batch conditions; actual edge-device suitability untested

### 5.3 Screening Outcomes and Class-Level Errors

- ResNet frozen screening: 157 missed anomalous images / 1,492; 71 false alarms / 1,496 normal images
- Explicit denominators; recall versus normal false-alarm rate; operational error costs
- Test confusion matrix and weak-class table: Soiling F1 0.3125, Hot-Spot 0.3750, Cell-Multi 0.4248
- Support counts and rare-class uncertainty; recurring subtype confusions
- Offline-Module → No-Anomaly: 49/123 on standard-ResNet validation; validation/test and model-version labels

### 5.4 Uncertainty and Statistical Interpretation

- Individual model intervals: ResNet macro-F1 0.5939–0.6515; CNN 0.5660–0.6237
- Paired difference interval: +0.0020 to +0.0524; observed gap +0.0275
- Positive conditional benchmark gap; overlapping marginal intervals versus paired uncertainty
- Post-hoc timing; single-checkpoint comparison; training-seed uncertainty outside the analysis
- Detected related-image groups versus independent sites; limited external validity

### 5.5 Explanation and Crop-Sensitivity Evidence

- Selected Grad-CAM cases: correct predictions, missed fault and false alarm; short captions and empty-map example
- Coarse 3 × 2 maps; class-dependent contributions; qualitative interpretation limits
- Robustness table/plot: original macro-F1 0.6281; shifts 0.5550–0.5925; rotations approximately 0.567
- Half resolution: macro-F1 0.5036, screening recall 83.38%; retained-detail sensitivity
- Blur trade-off; unchanged labels; synthetic padding and lost evidence as alternative contributors to score changes

### 5.6 Issues Faced and Limitations

- Small image size, class imbalance and ambiguous thermal patterns
- Repeated validation selection and mostly single-seed experiments
- Missing site/camera/capture identifiers; unverified transfer to new inspection settings
- Synthetic disturbances versus real capture/cropping errors; no complete-system accuracy measurement
- Issue → response → remaining limitation; distinction between resolved implementation issues and unresolved scientific limits

## 6. Potential for Wider Adoption

### 6.1 Proposed End-to-End Inspection Workflow

- Thermal capture → panel detection → crop/alignment → quality checks → classification/screening → technician review
- Current contribution: crop classifier and recorded notebook demonstration
- Required additions: detector/cropper, capture validation, panel identity tracking and review interface
- Qualified physical/electrical assessment before repair, reuse or disposal decisions

### 6.2 Stakeholder-Specific Recommendations

- Technicians: inspection prioritisation; visible uncertainty/limitations; human review of flagged and unsuitable inputs
- Operators: missed-fault costs, inspection capacity and prospective workflow evaluation
- Refurbishers/recyclers: thermal screening alongside physical/electrical testing; avoidance of automatic disposal/clearance
- Purchasers/installers/communities: accountability for safety and reuse decisions; traceable escalation
- SmallCNN as a lower-compute candidate; device-specific latency, energy and operational testing before selection

### 6.3 Impact, Ethics and Socio-Technical Risks

- False negatives: delayed investigation; false positives: unnecessary inspection or premature rejection
- Uneven class reliability and representation; module technologies and sites missing verified coverage
- Automation bias; uncalibrated scores; Grad-CAM overinterpretation; clear responsibility for final decisions
- Site-data permissions, identifying metadata and secure operational records
- Potential waste/inspection benefits versus unmeasured environmental savings; compute footprint, training needs and access barriers

### 6.4 Prioritised Future Work

- First: independent site-labelled evaluation; representative capture conditions and operational fault prevalence
- Next: crop/detection quality, image-quality flags, human-review workflow and full-pipeline failure analysis
- Then: targeted label review, repeated seeds and economical model improvements; fresh evaluation for revised policies
- Prospective stakeholder testing; measured inspection cost, device energy and reuse outcomes

## 7. Conclusions

### Overall Outcome and Next Decision

- Achievements against the confirmed proposal; completed classifier, reproducible evidence and notebook demonstration
- Positive ResNet benchmark gap; missed research targets; substantial compute trade-off
- Crop/detail sensitivity and unresolved external validity
- Qualified technician-led triage as intended direction; independent pipeline validation as the next priority

## References

### Citation Coverage

- Dataset repository/paper; ResNet/transfer-learning primary work; pretrained-weight documentation; Grad-CAM paper
- Verified PV inspection, reuse/waste and full-pipeline sources; source-specific evidence for factual claims
- Consistent numbered citation style matching the LaTeX template; figures/tables with provenance and split labels
- Relevant comparison conditions; source verification before numerical cross-study claims

## Appendices

### Appendix A — Extended Results and Supporting Figures

- Per-class tables, complete confusion matrices, extra Grad-CAM cases and robustness condition definitions
- Detailed development records, negative results and benchmark scope; core interpretation retained in the main body

### Appendix B — Reproducibility and Repository Guide

- Public repository link and final reviewed commit; notebook order and evidence-file map
- Data/licence setup, environment, retained artifact requirements and deliberate reproduction commands
- Protocol/source/checkpoint fingerprints; original frozen evaluation versus subsequent saved-prediction analysis

### Appendix C — Team Contributions

- Iris Li, Ethan Melton, Anushka Ram and Tom Wait: membership confirmation against the submitted proposal
- Actual responsibilities, outputs and cross-review evidence; planned roles distinguished from completed work
- Contributions beyond Git commits; team-agreed attribution and SPARKplus consistency

### Appendix D — Generative AI Use

- Original substantive prompts and outputs; tool/model/date where verifiable
- Generated code, interpretation and writing assistance; accepted, modified and rejected suggestions
- Team modifications, independent checks and automated verification evidence
- [Working AI-use record](ai_usage.md); original conversation retention; completion of the required report appendix

## Evidence Map — Writing Reference, Outside the Report Body

### Primary Repository Evidence

- [Notebook 01](../notebooks/01_data_exploration.ipynb): dataset properties, imbalance and cleaning
- [Notebook 02](../notebooks/02_baseline_experiments.ipynb): CNN choices, errors and retained reference
- [Notebook 03](../notebooks/03_resnet18_experiments.ipynb): progression, compute and sections 8–10
- [Frozen final results](../results/final-evaluation-v1.md) and [paired comparison](../results/final-model-comparison-v1.md): final outcomes and conditional uncertainty
- [Crop-robustness record](../results/resnet18-validation-robustness-v1.json) and [contribution worksheet](contributions.md): sensitivity evidence and attribution preparation
