"""Export the two verified early-downsampling experiments without dataset inference."""
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
from summarise_resnet_generalisation import checked_run,read
ROOT=Path(__file__).resolve().parents[1]


def main():
    protocol=read('configs/resnet18-spatial-v1.json')
    ledger=read('results/resnet18-spatial-v1-ledger.json')
    if ledger['status']!='complete_verified': raise ValueError('Finish and verify both experiments before exporting outcomes.')
    benchmark=read('results/resnet18-spatial-v1-inference.json')
    reference=checked_run(protocol['reference_run'],'standard')
    trials=[checked_run(t['run_id'],t['arm'],t['plan']) for t in protocol['planned_trials']]
    runs=[reference]+trials
    controlled=('cpu_threads','device','parent_model','manifest_sha256','parent_checkpoint_sha256','pretrained_sha256','seed','batch_size','head_learning_rate','layer4_learning_rate','max_epochs','early_stopping_patience','class_weighting','class_weight_exponent','label_smoothing','optimizer','weight_decay','train_augmentation','scheduler','scheduler_settings')
    parent_cost=read('runs/resnet18-frozen-repro-s2112-v1/summary.json')['elapsed_seconds']/60
    rows=[]
    for run in runs:
        p,summary,m=(run[k] for k in ('plan','summary','metrics'))
        for key in controlled:
            if p.get(key)!=reference['plan'].get(key): raise ValueError('Control differs: '+key)
        variant=p.get('spatial_variant','standard')
        if run['verification']['spatial_variant']!=variant: raise ValueError('Verification restored a different spatial variant.')
        bench=next(b for b in benchmark['models'] if b['run_id']==run['run_id'])
        if bench['checkpoint_sha256']!=run['verification']['checkpoint_sha256']: raise ValueError('Benchmark used different checkpoint weights.')
        rows.append(dict(variant=variant,run_id=run['run_id'],macro_f1=m['macro_f1'],accuracy=m['accuracy'],balanced_accuracy=m['balanced_accuracy'],missed_faults=m['faults_called_normal'],false_alarms=m['normal_called_fault'],best_epoch=summary['best_epoch'],epochs_completed=summary['epochs_completed'],stop_reason=summary['stop_reason'],fit_minutes=summary['elapsed_seconds']/60,workflow_minutes=summary['elapsed_seconds']/60+parent_cost,mean_epoch_seconds=summary['elapsed_seconds']/len(run['history']),trainable_parameters=run['provenance']['trainable_parameters'],total_parameters=run['provenance']['total_parameters'],delta_macro_f1=m['macro_f1']-reference['metrics']['macro_f1'],batch_one_ms=bench['batches'][0]['median_batch_ms'],batch64_ms=bench['batches'][1]['median_batch_ms'],feature_geometry=bench['geometry']))
    selected=max(runs,key=lambda r:r['metrics']['macro_f1'])
    snapshot=dict(protocol=protocol['protocol'],created_utc=datetime.now(timezone.utc).isoformat(),reference_run=reference['run_id'],selected_run=selected['run_id'],selection_metric='validation_macro_f1',runs=runs,comparison=rows,new_training_minutes=sum(r['summary']['elapsed_seconds'] for r in trials)/60,inference_benchmark='results/resnet18-spatial-v1-inference.json',test_evaluated=False,scope_note=protocol['scope_note'],replication=protocol['replication'],controlled_settings_verified=list(controlled))
    # Review remaining validation errors without running any additional inference.
    sys.path.insert(0,str(ROOT/'src'))
    from pv_fault.data import CLASS_NAMES
    metrics=selected['metrics']
    weakest=sorted(metrics['per_class'],key=lambda label: metrics['per_class'][label]['f1'])[:3]
    confusions=[]
    for i,row in enumerate(metrics['confusion_matrix']):
        for j,count in enumerate(row):
            if i!=j and count:
                confusions.append(dict(true_label=CLASS_NAMES[i],predicted_label=CLASS_NAMES[j],count=count,share_of_true_class=count/sum(row)))
    records=read(Path('runs')/selected['run_id']/'best_validation_predictions.json')
    examples=[]
    for label in weakest:
        wrong=sorted([r for r in records if CLASS_NAMES[r['true_index']]==label and r['true_index']!=r['predicted_index']],key=lambda r:int(r['image_id']))[:2]
        for record in wrong:
            examples.append(dict(image_id=record['image_id'],image_path=record['image_path'],true_label=label,predicted_label=CLASS_NAMES[record['predicted_index']],predicted_probability=record['probabilities'][record['predicted_index']]))
    snapshot['error_review']=dict(selected_run=selected['run_id'],weakest_classes={label:metrics['per_class'][label] for label in weakest},top_confusions=sorted(confusions,key=lambda r:-r['count'])[:10],examples=examples,selection_rule='Three lowest-F1 classes; up to two wrong predictions per class, smallest numeric image IDs; descriptive validation review only.')
    (ROOT/'results/resnet18-spatial-v1.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    lines=['# ResNet early-downsampling experiments','',protocol['controls'],'',
           '| Variant | Macro-F1 | Accuracy | Balanced accuracy | Missed faults / 1492 | False alarms / 1496 | Best/completed epochs | Fit min | Median batch-one ms | Median batch-64 ms |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['variant']} | {row['macro_f1']:.4f} | {row['accuracy']:.2%} | {row['balanced_accuracy']:.2%} | {row['missed_faults']} | {row['false_alarms']} | {row['best_epoch']}/{row['epochs_completed']} | {row['fit_minutes']:.2f} | {row['batch_one_ms']:.3f} | {row['batch64_ms']:.3f} |")
    lines += ['',f"Selected by validation macro-F1: `{selected['run_id']}`. The standard augmented reference remains eligible; a larger grid is not automatically adopted.",'',
              'Stride1 retains the 7×7 kernel tensors and changes convolution stride to one. No_maxpool retains convolution stride two and replaces only the initial max pool with Identity. Both preserve a 512×5×3 layer4 grid versus 512×3×2 for standard ResNet. Global average pooling and the classifier are unchanged. Total and trainable parameter counts are unchanged; activations and operations increase.', '',
              f"The two new full fits used {snapshot['new_training_minutes']:.2f} local CPU wall minutes in total, excluding smoke checks, external checkpoint verification, benchmarks and reporting. Each includes initialization, training with loading, two validation passes, metrics and saving. A fresh workflow also includes the reused frozen-head parent ({parent_cost:.2f} recorded minutes); ImageNet pretraining cost is excluded. All runs used four CPU threads. Local load varies; historical fit times were not interleaved repeated timing trials.", '',
              'Actual selected checkpoints were benchmarked after all training on identical synthetic [N,1,40,24] inputs, with five warmups and twenty measured forwards per batch size. No dataset images were loaded for the benchmark. Raw timings, source/checkpoint hashes, feature geometries and SmallCNN measurements are retained separately. Parameter storage is not peak RAM; no FLOPs or energy were measured.', '',
              '## Interpretation and remaining work','',protocol['scope_note'],'',
              'Each architecture was applied before any updates in its independent run, starting from the same frozen-head checkpoint as the augmented reference. The reference is not a warm-start parent for these comparisons. Larger early grids change activation distributions even with identical pretrained tensors, and fixed BatchNorm statistics may become less suitable. A result here cannot establish whether the variant would help with stem-specific head refitting or broader adaptation.', '',
              'Full per-class precision/recall/F1, confusion matrices and epoch histories are retained in the JSON and notebook. Checkpoints reproduced all 2,988 validation predictions including probabilities. Variant metadata is explicitly restored during verification because stride and pooling are absent from learned state dictionaries. One seed only; repeated validation selection can overestimate gains. Additional seeds remain deferred, and original-test inference remains unused.']
    a,b,c=rows
    lines += ['', '## Decision after both fits', '',
              f"Stride1 increased validation macro-F1 by {b['delta_macro_f1']:+.4f}, from {a['macro_f1']:.4f} to {b['macro_f1']:.4f}. Accuracy changed from {a['accuracy']:.2%} to {b['accuracy']:.2%}, while missed faults / false alarms changed from {a['missed_faults']} / {a['false_alarms']} to {b['missed_faults']} / {b['false_alarms']}. Balanced accuracy was essentially unchanged. This is a small single-seed gain, with worse missed-fault performance, not an improvement on every metric.", '',
              f"Stride1 took {b['fit_minutes']:.2f} fit minutes versus {a['fit_minutes']:.2f} recorded reference minutes. Its measured batch-one latency was {b['batch_one_ms']/a['batch_one_ms']:.2f} times the standard model, and batch-64 latency {b['batch64_ms']/a['batch64_ms']:.2f} times. Parameter count stayed fixed; preserving more spatial activations increased actual compute.", '',
              f"Removing max pooling did not beat the reference on macro-F1 ({c['macro_f1']:.4f}), although it reduced missed faults to {c['missed_faults']} and increased balanced accuracy to {c['balanced_accuracy']:.2%}. False alarms rose to {c['false_alarms']}. It therefore represents another error trade-off, not a uniform decline on every metric.", '',
              'Keep the standard augmented reference as the lower-cost option and stride1 as the highest-macro-F1 candidate for later replication. Do not automatically replace the model based on this small validation gain. The deterministic weak-class review and saved example images in the notebook should guide any next hypothesis. A spatial-grid classifier, stem-specific head refit, BatchNorm adaptation and additional seeds were not run in this batch.']
    (ROOT/'results/resnet18-spatial-v1.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(selected_run=selected['run_id'],comparison=rows,new_training_minutes=snapshot['new_training_minutes']),indent=2))

if __name__=='__main__': main()
