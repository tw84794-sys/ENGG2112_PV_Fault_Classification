"""Post-hoc paired group bootstrap of saved test predictions; no model inference."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pv_fault.data import CLASS_NAMES, CLASS_TO_INDEX
from pv_fault.evaluate import classification_metrics
from pv_fault.train import file_hash

RESULT = ROOT / 'results/final-model-comparison-v1.json'
REPORT = RESULT.with_suffix('.md')


def paired_gap(cnn, resnet, *, resamples, seed, confidence):
    if not cnn or len(cnn) != len(resnet) or resamples < 1 or not 0 < confidence < 1:
        raise ValueError('Expected nonempty paired predictions and valid bootstrap settings.')
    keys = ('image_id', 'image_path', 'group_id', 'true_index')
    if len({r['image_id'] for r in cnn}) != len(cnn):
        raise ValueError('Repeated image identity.')
    if any(not a['group_id'] or any(a[key] != b[key] for key in keys) for a, b in zip(cnn, resnet)):
        raise ValueError('Models must describe the same images, labels and groups in the same order.')
    truth = np.asarray([r['true_index'] for r in cnn])
    predictions = [np.asarray([r['predicted_index'] for r in rows]) for rows in (cnn, resnet)]
    observed = [classification_metrics(truth, p)['macro_f1'] for p in predictions]
    groups, inverse = np.unique([r['group_id'] for r in cnn], return_inverse=True)
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(resamples):
        counts = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups))
        indices = np.repeat(np.arange(len(cnn)), counts[inverse])
        scores = [classification_metrics(truth[indices], p[indices])['macro_f1'] for p in predictions]
        gaps.append(scores[1] - scores[0])
    tail = (1 - confidence) / 2
    return {'direction': 'ResNet18 minus SmallCNN test macro-F1', 'observed': observed[1] - observed[0],
            'lower': float(np.quantile(gaps, tail)), 'upper': float(np.quantile(gaps, 1 - tail)),
            'groups': len(groups), 'resamples': resamples, 'seed': seed, 'confidence': confidence,
            'method': 'Paired percentile bootstrap of the same recorded related-image groups'}


def main():
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError('Preserve the existing comparison record; refusing to overwrite it.')
    evaluation_path = ROOT / 'results/final-evaluation-v1.json'
    protocol_path = ROOT / 'configs/final-evaluation-v1.json'
    evaluation = json.loads(evaluation_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    completed = json.loads((ROOT / 'runs/final-evaluation-v1/completed.json').read_text())
    if (evaluation['split'] != 'test' or not evaluation['test_evaluated']
            or evaluation['protocol_sha256'] != file_hash(protocol_path)
            or completed['results_sha256'] != file_hash(evaluation_path)
            or tuple(protocol['class_names']) != CLASS_NAMES):
        raise ValueError('Frozen test record or protocol differs.')
    sources = ('src/pv_fault/evaluate.py', 'src/pv_fault/data.py')
    for name in sources:
        if file_hash(ROOT / name) != protocol['sources'][name]:
            raise ValueError(f'Frozen metric/data source changed: {name}')
    manifest = ROOT / 'data/splits/infrared_solar_modules_v1.csv'
    if file_hash(manifest) != protocol['dataset']['manifest_sha256']:
        raise ValueError('Frozen manifest changed.')
    with manifest.open(newline='') as source:
        test_rows = [r for r in csv.DictReader(source) if r['split'] == 'test']
    records = []
    for name in ('SmallCNN', 'ResNet18'):
        path = ROOT / 'runs/final-evaluation-v1' / (name + '_predictions.json')
        if file_hash(path) != evaluation['models'][name]['predictions_sha256']:
            raise ValueError(f'Saved predictions changed: {name}')
        rows = json.loads(path.read_text())
        if len(rows) != len(test_rows):
            raise ValueError('Predictions do not cover the frozen test split.')
        for row, expected in zip(rows, test_rows):
            if (any(row[key] != expected[key] for key in ('image_id', 'image_path', 'group_id'))
                    or row['true_index'] != CLASS_TO_INDEX[expected['label']]):
                raise ValueError('Prediction identity differs from the frozen test manifest.')
        metrics = classification_metrics([r['true_index'] for r in rows], [r['predicted_index'] for r in rows])
        if metrics != evaluation['models'][name]['multiclass']:
            raise ValueError('Saved predictions do not reproduce frozen metrics.')
        records.append(rows)
    gap = paired_gap(*records, **protocol['bootstrap'])
    interpretation = ('The interval is above zero, supporting a positive macro-F1 gap on this benchmark.'
                      if gap['lower'] > 0 else
                      'The interval includes zero or lies below it; a positive macro-F1 gap is not established.')
    result = {'analysis': 'Post-hoc comparison of the two previously frozen models', 'split': 'test',
              'new_model_inference': False, 'model_selection_performed': False, 'images': len(test_rows),
              'evaluation_results_sha256': file_hash(evaluation_path), 'protocol_sha256': file_hash(protocol_path),
              'prediction_sha256': {name: evaluation['models'][name]['predictions_sha256']
                                    for name in ('SmallCNN', 'ResNet18')},
              'source_sha256': {name: file_hash(ROOT / name) for name in (*sources, 'scripts/compare_frozen_models.py')},
              'numpy_version': np.__version__, 'paired_gap': gap, 'interpretation': interpretation,
              'limitations': 'Added after observing final test summaries. Conditions on these trained checkpoints '
                             'and detected related-image groups; no uncertainty across training seeds or verified '
                             'independent sites. Does not replace the original frozen evaluation or justify retuning.'}
    with RESULT.open('x') as target:
        target.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    text = f'''# Paired comparison of frozen models

The observed test macro-F1 difference, **ResNet18 − SmallCNN**, is **{gap['observed']:.4f}**.
Its 95% paired group-bootstrap interval is **{gap['lower']:.4f} to {gap['upper']:.4f}**.

{interpretation}

The analysis reuses saved predictions for all {len(test_rows):,} test images. Each of {gap['resamples']:,} resamples draws the same related-image groups for both models, preserving within-group rows and pairing. Settings match the original bootstrap: seed {gap['seed']}, 95% percentile interval. No checkpoints or images are loaded and no inference, fitting or threshold selection occurs.

This analysis was added after observing the final test summaries. It conditions on the two frozen checkpoints and detected related-image groups. It cannot establish superiority across training seeds, independent sites, deployment hardware or operational objectives. A positive interval would still not meet the proposal's +0.05 improvement target when the observed gap is only +0.0275.

The [original evaluation](final-evaluation-v1.md) remains unchanged. Prediction and source fingerprints are retained in the [comparison record](final-model-comparison-v1.json); full saved predictions remain in the ignored local runs folder. Regeneration requires those prediction files, not model checkpoints.
'''
    with REPORT.open('x') as target:
        target.write(text)
    print(text)


if __name__ == '__main__':
    main()
