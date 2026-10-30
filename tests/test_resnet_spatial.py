"""Spatial changes preserve pretrained tensors and restore distinct forward paths."""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pv_fault.models.resnet import ResNet18Thermal
from pv_fault.finetuning import configure_resnet_finetuning
from pv_fault.resnet_spatial import configure_spatial_variant

class SpatialVariantTests(unittest.TestCase):
    def make_model(self):
        path=ROOT/'data/pretrained/resnet18-f37072fd.pth'
        if not path.exists(): self.skipTest('Verified pretrained weights not downloaded locally')
        return ResNet18Thermal(12,path)

    def test_geometry_changes_without_changing_pretrained_tensors_or_parameters(self):
        torch.set_num_threads(4)
        model=self.make_model()
        original={k:v.clone() for k,v in model.state_dict().items()}
        count=sum(p.numel() for p in model.parameters())
        images=torch.rand(2,1,40,24)
        for variant,conv_shape,layer_shape in [('standard',(40,24),(3,2)),('stride1',(80,48),(5,3)),('no_maxpool',(40,24),(5,3))]:
            with self.subTest(variant=variant):
                configure_spatial_variant(model,variant)
                self.assertEqual(count,sum(p.numel() for p in model.parameters()))
                self.assertEqual(set(original),set(model.state_dict()))
                self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in original.items()))
                shapes={}
                def hook(name):
                    def capture(module,args,output): shapes[name]=tuple(output.shape[-2:])
                    return capture
                handles=[model.backbone.conv1.register_forward_hook(hook('conv1')),model.backbone.layer4.register_forward_hook(hook('layer4'))]
                with torch.inference_mode(): output=model(images)
                for handle in handles: handle.remove()
                self.assertEqual(shapes['conv1'],conv_shape)
                self.assertEqual(shapes['layer4'],layer_shape)
                self.assertEqual(tuple(output.shape),(2,12))
                self.assertTrue(torch.isfinite(output).all())

    def test_each_variant_keeps_bn_frozen_and_late_stage_gradients_finite(self):
        torch.set_num_threads(4)
        for variant in ('stride1','no_maxpool'):
            with self.subTest(variant=variant):
                model=self.make_model()
                configure_spatial_variant(model,variant)
                configure_resnet_finetuning(model)
                model.train()
                self.assertFalse(any(m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d)))
                nn.CrossEntropyLoss()(model(torch.rand(2,1,40,24)),torch.tensor([0,7])).backward()
                self.assertIsNone(model.backbone.conv1.weight.grad)
                self.assertTrue(torch.isfinite(model.backbone.layer4[0].conv1.weight.grad).all())
                self.assertTrue(torch.isfinite(model.backbone.fc.weight.grad).all())

    def test_unknown_variant_rejected_before_model_mutation(self):
        with self.assertRaises(ValueError): configure_spatial_variant(None,'both_changes')

if __name__=='__main__': unittest.main()
