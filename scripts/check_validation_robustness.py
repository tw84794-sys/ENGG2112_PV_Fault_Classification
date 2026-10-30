"""Fixed synthetic disturbances of validation crops; no training or test inference."""
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_final import committed_protocol, load_model, write_new
from pv_fault.data import SolarModuleDataset
from pv_fault.evaluate import classification_metrics, collect_validation_predictions
from pv_fault.final_evaluation import screening_metrics
from pv_fault.train import file_hash

# Fixed before scoring; magnitudes describe simulations, not camera requirements.
CONDITIONS = {
    'original': 'Unchanged 40 × 24 crop',
    'shift_left': 'Shift left 2 pixels; replicate edges',
    'shift_right': 'Shift right 2 pixels; replicate edges',
    'shift_up': 'Shift up 2 pixels; replicate edges',
    'shift_down': 'Shift down 2 pixels; replicate edges',
    'rotate_minus5': 'Rotate −5°; bilinear; replicate-edge padding',
    'rotate_plus5': 'Rotate +5°; bilinear; replicate-edge padding',
    'blur': 'Gaussian blur: 3 × 3 kernel, sigma 0.75 pixels',
    'half_resolution': '40 × 24 → 20 × 12 → 40 × 24; bilinear, antialiased',
}
RESULTS = ROOT / 'results/resnet18-validation-robustness-v1.json'
RUN = ROOT / 'runs/resnet18-validation-robustness-v1'


def disturb(image, condition):
    if image.shape != (1, 40, 24) or not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError('Expected one finite floating-point grayscale crop [1, 40, 24].')
    if condition not in CONDITIONS:
        raise ValueError('Unknown disturbance.')
    if condition == 'original':
        return image
    shifts = {'shift_left': (-2, 0), 'shift_right': (2, 0),
              'shift_up': (0, -2), 'shift_down': (0, 2)}
    if condition in shifts:
        dx, dy = shifts[condition]
        padded = TF.pad(image, [2, 2, 2, 2], padding_mode='edge')
        return padded[:, 2 - dy:42 - dy, 2 - dx:26 - dx]
    if condition.startswith('rotate_'):
        padded = TF.pad(image, [4, 4, 4, 4], padding_mode='edge')
        rotated = TF.rotate(padded, -5 if condition == 'rotate_minus5' else 5,
                            interpolation=InterpolationMode.BILINEAR)
        return TF.center_crop(rotated, [40, 24])
    if condition == 'blur':
        return TF.gaussian_blur(image, kernel_size=3, sigma=.75)
    small = TF.resize(image, [20, 12], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.resize(small, [40, 24], interpolation=InterpolationMode.BILINEAR, antialias=True)


class DisturbedValidation(Dataset):
    def __init__(self, dataset, condition):
        if dataset.split != 'val' or condition not in CONDITIONS:
            raise ValueError('Choose a declared disturbance of validation data only.')
        self.dataset, self.condition = dataset, condition
        self.rows, self.split = dataset.rows, 'val'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return disturb(image, self.condition), label


def main():
    if RESULTS.exists() or RUN.exists():
        raise FileExistsError('Preserve the existing robustness record; refusing to overwrite it.')
    protocol = committed_protocol()
    spec = next(s for s in protocol['models'] if s['name'] == 'ResNet18')
    torch.set_num_threads(protocol['cpu_threads'])
    model = load_model(spec)
    dataset = SolarModuleDataset('val')
    threshold = spec['screening']['threshold']
    sources = ['scripts/check_validation_robustness.py', *protocol['sources']]
    result = {'split': 'val', 'test_evaluated': False, 'training_performed': False,
              'protocol_sha256': file_hash(ROOT / 'configs/final-evaluation-v1.json'),
              'model': spec, 'images_per_condition': len(dataset),
              'screening_threshold': threshold, 'batch_size': protocol['batch_size'],
              'cpu_threads': torch.get_num_threads(), 'environment': protocol['environment'],
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
              'source_sha256': {name: file_hash(ROOT / name) for name in sources},
              'conditions': CONDITIONS, 'results': [],
              'limitations': 'Exploratory synthetic validation stress test after final evaluation; not model selection. '
                             'Each disturbance is applied independently at one declared severity. '
                             'Edge replication invents border pixels; shifts remove some original content. '
                             'No real camera, detection, segmentation or site generalisation is established.'}
    RUN.mkdir(exist_ok=False)
    write_new(RUN / 'plan.json', result)
    started = time.perf_counter()
    for condition in CONDITIONS:
        loader = DataLoader(DisturbedValidation(dataset, condition),
                            batch_size=protocol['batch_size'], shuffle=False)
        records = collect_validation_predictions(model, loader, 'cpu')
        if condition == 'original':
            reference = json.loads((ROOT / 'runs' / spec['run_id'] / 'best_validation_predictions.json').read_text())
            if records != reference:
                raise ValueError('Unchanged validation predictions differ from the frozen reference.')
            result['original_predictions_verified'] = True
        metrics = classification_metrics([r['true_index'] for r in records], [r['predicted_index'] for r in records])
        binary = screening_metrics(records, threshold)
        path = RUN / f'{condition}.json'
        write_new(path, records)
        row = {'condition': condition, 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
               'screening': binary, 'predictions_sha256': file_hash(path)}
        result['results'].append(row)
        print(f"{condition}: macro-F1 {row['macro_f1']:.4f}, recall {binary['recall']:.2%}, false alarms {binary['false_alarm_rate']:.2%}", flush=True)
    result['elapsed_seconds'] = time.perf_counter() - started
    if any(file_hash(ROOT / name) != expected for name, expected in spec['files'].items()):
        raise ValueError('A frozen model input changed during the check.')
    write_new(RESULTS, result)
    print(f'Saved {RESULTS}', flush=True)


if __name__ == '__main__':
    main()
