"""Export verified local ResNet generalisation results; never load dataset images."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads((ROOT / path).read_text())

def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def checked_run(run_id, arm, plan_path=None):
    folder = Path('runs') / run_id
    plan = read(folder / 'experiment-plan.json')
    provenance = read(folder / 'provenance.json')
    summary = read(folder / 'summary.json')
    metrics = read(folder / 'best_validation_metrics.json')
    verification = read(folder / 'checkpoint_verification.json')
    history = read(folder / 'history.json')
    if summary['test_evaluated'] or provenance['test_evaluated'] or verification['test_evaluated']:
        raise ValueError('Test inference is outside this experiment.')
    if not verification['checkpoint_predictions_verified'] or verification['validation_images'] != 2988:
        raise ValueError('Checkpoint validation predictions are not verified.')
    if verification['checkpoint_sha256'] != digest(folder / 'best_model.pt'):
        raise ValueError('Checkpoint has changed since verification.')
    if plan_path and (plan != read(plan_path) or provenance['plan_sha256'] != digest(plan_path)):
        raise ValueError('Saved and tracked experiment plans differ.')
    for source, expected in provenance['source_sha256'].items():
        if digest(folder / 'source_snapshot' / source) != expected:
            raise ValueError('An archived producer source has changed.')
    if summary['best_validation_macro_f1'] != metrics['macro_f1'] or metrics['count'] != 2988:
        raise ValueError('Saved summary and metrics disagree.')
    if sum(map(sum, metrics['confusion_matrix'])) != 2988:
        raise ValueError('Confusion matrix has wrong total.')
    best = next(row for row in history if row['epoch'] == summary['best_epoch'])
    if best['validation_macro_f1'] != metrics['macro_f1']:
        raise ValueError('Best history row differs from metrics.')
    return dict(arm=arm, run_id=run_id, plan=plan, provenance=provenance, summary=summary,
                metrics=metrics, history=history, verification=verification)

def main():
    protocol = read('configs/resnet18-generalisation-v1.json')
    ledger = read('results/resnet18-generalisation-v1-ledger.json')
    if ledger['status'] != 'complete_verified':
        raise ValueError('Finish and verify all planned trials before exporting a final comparison.')
    reference = checked_run(protocol['reference_run'], 'reference')
    candidates = [checked_run(t['run_id'], t['arm'], t['plan']) for t in protocol['planned_trials']]
    all_runs = [reference] + candidates
    for run in all_runs:
        if run['plan']['manifest_sha256'] != reference['plan']['manifest_sha256']:
            raise ValueError('Comparison uses inconsistent splits.')
    # Older producer records omit total_parameters; derive it from the shared
    # verified architecture without modifying their historical provenance.
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from pv_fault.models.resnet import ResNet18Thermal
    torch.set_num_threads(4)
    architecture = ResNet18Thermal(12, ROOT / reference['plan']['pretrained_weights'], reference['plan']['pretrained_sha256'])
    total_parameters = sum(p.numel() for p in architecture.parameters())
    del architecture
    for run in all_runs:
        recorded = run['provenance'].get('total_parameters', total_parameters)
        if recorded != total_parameters:
            raise ValueError('The forward architecture changed between arms.')
    winner = max(all_runs, key=lambda r: r['metrics']['macro_f1'])
    frozen_cost = read('runs/resnet18-frozen-repro-s2112-v1/summary.json')['elapsed_seconds']
    rows = []
    for run in all_runs:
        plan, summary, metrics = (run[k] for k in ('plan','summary','metrics'))
        warm = plan.get('parent_model', 'resnet18_frozen') == 'resnet18_layer4'
        inherited = frozen_cost + (reference['summary']['elapsed_seconds'] if warm else 0)
        rows.append(dict(arm=run['arm'],run_id=run['run_id'],macro_f1=metrics['macro_f1'],
                         accuracy=metrics['accuracy'],balanced_accuracy=metrics['balanced_accuracy'],
                         missed_faults=metrics['faults_called_normal'],false_alarms=metrics['normal_called_fault'],
                         best_epoch=summary['best_epoch'],epochs_completed=summary['epochs_completed'],
                         stop_reason=summary['stop_reason'],incremental_minutes=summary['elapsed_seconds']/60,
                         cumulative_workflow_minutes=(summary['elapsed_seconds']+inherited)/60,
                         inherited_training_minutes=inherited/60,
                         trainable_parameters=run['provenance']['trainable_parameters'],
                         total_parameters=total_parameters,
                         sum_training_seconds=(sum(h['train_seconds'] for h in run['history']) if all('train_seconds' in h for h in run['history']) else None),
                         sum_validation_seconds=(sum(h['validation_loss_seconds']+h['validation_prediction_seconds'] for h in run['history']) if all('validation_loss_seconds' in h and 'validation_prediction_seconds' in h for h in run['history']) else None),
                         delta_macro_f1_vs_reference=metrics['macro_f1']-reference['metrics']['macro_f1']))
    warm_control = next(r for r in candidates if r['arm'] == 'continue4')
    gradual = next(r for r in candidates if r['arm'] == 'gradual34')
    snapshot = dict(protocol=protocol['protocol'],created_utc=datetime.now(timezone.utc).isoformat(),
                    reference_run=reference['run_id'],selected_run=winner['run_id'],selection_metric='validation_macro_f1',
                    runs=all_runs,comparison=rows,gradual_minus_matched_continuation_macro_f1=gradual['metrics']['macro_f1']-warm_control['metrics']['macro_f1'],
                    new_training_wall_minutes=sum(r['summary']['elapsed_seconds'] for r in candidates)/60,
                    timing_scope=protocol['timing_scope'],test_evaluated=False,
                    limitations=['One seed only; repeated validation selection can overestimate generalisation.',
                                 'Ten-epoch fresh arms and four-epoch continuations have different histories; compare stage unfreezing with its matched continuation.',
                                 'Inherited training is reused across trials; per-workflow cumulative totals should not be summed as actual search cost.',
                                 'No peak RAM, energy or FLOPs measured. No new inference benchmark was run; all arms retain the same forward architecture.',
                                 'Recorded local CPU wall times vary with machine load; historical reference and new fits were not interleaved repeated timing trials.',
                                 'Augmented training accuracy and differently weighted/smoothed training losses are not directly comparable.',
                                 'Any selected validation improvement requires later replication; original test remains reserved.'])
    target = ROOT / 'results/resnet18-generalisation-v1.json'
    target.write_text(json.dumps(snapshot,indent=2)+'\n')
    lines = ['# ResNet generalisation experiments','',
             'Five independent local CPU comparisons against the verified scheduled layer4 reference. The split, seed 2112, class order and test boundary are unchanged. Plans were fixed before full fits; each checkpoint was reloaded and reproduced all 2,988 validation predictions, including probabilities.','',
             '| Arm | Macro-F1 | Accuracy | Balanced accuracy | Missed faults | False alarms | Best/completed epochs | Fit min | Workflow min |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['macro_f1']:.4f} | {row['accuracy']:.2%} | {row['balanced_accuracy']:.2%} | {row['missed_faults']} | {row['false_alarms']} | {row['best_epoch']}/{row['epochs_completed']} | {row['incremental_minutes']:.2f} | {row['cumulative_workflow_minutes']:.2f} |")
    lines += ['',f"Selected by validation macro-F1: `{winner['run_id']}` ({winner['metrics']['macro_f1']:.4f}). The reference remains available even if a continuation performs worse.",'',
              f"Gradual layer3+4 minus matched layer4-only continuation: {snapshot['gradual_minus_matched_continuation_macro_f1']:+.4f} macro-F1. This paired comparison is the evidence for unfreezing another stage, rather than a comparison only with the older parent.",'',
              f"Actual new full-fit wall time across the five candidates: {snapshot['new_training_wall_minutes']:.2f} minutes, excluding smoke checks, verification and reporting. Each fit includes initialization, image loading, training, two validation passes, metrics and saving. Workflow minutes additionally include the reused frozen-parent fit and, for continuations, the reference fit. ImageNet pretraining cost is excluded. Do not sum per-workflow costs to estimate search compute.",'',
              'The input adapter and forward architecture are unchanged: 11,182,668 total parameters, with 8,394,764 trainable for layer4 or 10,491,916 for layer3+4. Prior architecture latency measurements still describe the compute scale; these new checkpoint weights were not benchmarked. No new claims about peak RAM, energy or FLOPs are made.','',
              '## Interpretation limits','']
    lines += ['- '+item for item in snapshot['limitations']]
    by_arm = {r['arm']: r for r in rows}
    aug, smooth, weight = (by_arm[a] for a in ('augmentation', 'smoothing', 'weight075'))
    lines += ['', '## What we learned', '',
              f"Mild augmentation was the best new candidate: macro-F1 {aug['macro_f1']:.4f}, a change of {aug['delta_macro_f1_vs_reference']:+.4f} versus the reference. This is a small single-seed improvement. Accuracy fell to {aug['accuracy']:.2%}; it missed {aug['missed_faults']} faults and produced {aug['false_alarms']} false alarms. Hot-Spot, Hot-Spot-Multi and Shadowing improved in F1, while Offline-Module and Vegetation worsened. Keep these limitations with the headline score.", '',
              f"Label smoothing 0.05 did not improve macro-F1 ({smooth['macro_f1']:.4f}), but changed the operating trade-off: {smooth['missed_faults']} missed faults and {smooth['false_alarms']} false alarms. It had the highest balanced accuracy among these candidates. It should not be described as uniformly worse on every metric.", '',
              f"The stronger class-weight exponent 0.75 was effectively tied in this run ({weight['macro_f1']:.4f}), with more false alarms than the reference. This does not establish that class-weight adjustment never helps; only one alternative exponent was tested.", '',
              f"Gradual layer3+4 tuning improved macro-F1 by only {snapshot['gradual_minus_matched_continuation_macro_f1']:+.4f} relative to its matched layer4 continuation. Both selected their first additional epoch and had the same fault/normal error counts. More trainable parameters and the inherited training cost did not beat augmentation. These runs reset optimiser state; they are not exact resumes.", '',
              'Retain the augmented candidate and the prior reference for later replication. Do not combine settings or treat this small validation gain as a confirmed test-set improvement. Additional seeds remain deferred, and original-test inference remains unused.', '',
              'Parameter totals for the old reference were reconstructed from the shared model architecture because its historical producer did not record that field. Missing old phase timings are represented as null, not zero.']
    (ROOT / 'results/resnet18-generalisation-v1.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(selected_run=winner['run_id'],comparison=rows,paired_stage_delta=snapshot['gradual_minus_matched_continuation_macro_f1']),indent=2))

if __name__ == '__main__':
    main()
