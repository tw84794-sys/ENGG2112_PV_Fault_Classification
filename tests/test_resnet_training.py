"""Training-only weights and explicit checkpoint lineage must remain valid."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pv_fault.data import CLASS_NAMES
from pv_fault.resnet_training import class_weights_from_training, validate_parent_metadata

class ResNetTrainingTests(unittest.TestCase):
    def dataset(self):
        return SimpleNamespace(split='train', rows=[{'label': name} for i,name in enumerate(CLASS_NAMES) for _ in range(i+1)])

    def test_weights_use_training_only_and_expected_rare_class_ratio(self):
        data=self.dataset()
        w=class_weights_from_training(data,.75)
        self.assertAlmostEqual(w.mean().item(),1.,places=6)
        self.assertAlmostEqual((w[0]/w[-1]).item(),12**.75,places=5)
        data.split='val'
        with self.assertRaises(ValueError): class_weights_from_training(data)
        with self.assertRaises(ValueError): class_weights_from_training(self.dataset(),1.1)

    def test_smoothing_reduces_gradient_towards_overconfident_correct_prediction(self):
        logits=torch.tensor([[6.,0.,0.]],requires_grad=True)
        unsmoothed=nn.CrossEntropyLoss()(logits,torch.tensor([0]))
        plain=torch.autograd.grad(unsmoothed,logits)[0]
        smoothed=nn.CrossEntropyLoss(label_smoothing=.05)(logits,torch.tensor([0]))
        smooth=torch.autograd.grad(smoothed,logits)[0]
        self.assertLess(plain[0,0],0)
        self.assertGreater(smooth[0,0],0)
        self.assertTrue(torch.isfinite(smooth).all())

    def test_warm_start_requires_explicit_matching_lineage(self):
        plan=dict(manifest_sha256='split',pretrained_sha256='weights')
        parent=dict(model='resnet18_layer4',manifest_sha256='split',class_names=CLASS_NAMES)
        provenance=dict(model='resnet18_layer4',manifest_sha256='split',pretrained_sha256='weights',test_evaluated=False)
        with self.assertRaises(ValueError): validate_parent_metadata(parent,plan,provenance)
        plan['parent_model']='resnet18_layer4'
        validate_parent_metadata(parent,plan,provenance)
        provenance['test_evaluated']=True
        with self.assertRaises(ValueError): validate_parent_metadata(parent,plan,provenance)
        provenance['test_evaluated']=False
        parent['manifest_sha256']='other'
        with self.assertRaises(ValueError): validate_parent_metadata(parent,plan,provenance)

if __name__=='__main__': unittest.main()
