import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_frozen_models import paired_gap


def rows(predicted):
    return [{'image_id': str(i), 'image_path': f'images/{i}.jpg', 'group_id': 'shared',
             'true_index': true, 'predicted_index': pred}
            for i, (true, pred) in enumerate(zip([0, 7], predicted))]


class PairedComparisonTests(unittest.TestCase):
    options = {'resamples': 40, 'seed': 2112, 'confidence': .95}

    def test_identical_models_have_zero_gap(self):
        result = paired_gap(rows([0, 7]), rows([0, 7]), **self.options)
        self.assertEqual([result[k] for k in ('observed', 'lower', 'upper')], [0, 0, 0])

    def test_group_members_stay_together_and_direction_is_correct(self):
        cnn, resnet = rows([7, 0]), rows([0, 7])
        result = paired_gap(cnn, resnet, **self.options)
        self.assertEqual(result['groups'], 1)
        for key in ('observed', 'lower', 'upper'):
            self.assertAlmostEqual(result[key], 1/6)
        reverse = paired_gap(resnet, cnn, **self.options)
        self.assertAlmostEqual(reverse['lower'], -result['upper'])
        self.assertEqual(result, paired_gap(cnn, resnet, **self.options))

    def test_misaligned_duplicate_or_missing_pairs_rejected(self):
        left = rows([0, 7])
        for right in (left[::-1], left[:1], []):
            with self.assertRaises(ValueError):paired_gap(left, right, **self.options)
        duplicate = [left[0], left[0]]
        with self.assertRaises(ValueError):paired_gap(duplicate, duplicate, **self.options)
        changed = rows([0, 7]);changed[0]['true_index'] = 1
        with self.assertRaises(ValueError):paired_gap(left, changed, **self.options)
        with self.assertRaises(ValueError):paired_gap([], [], **self.options)
