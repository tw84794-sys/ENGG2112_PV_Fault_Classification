"""Benchmark actual spatial checkpoints on identical synthetic inputs after fits."""
import json
import platform
import sys
from datetime import datetime,timezone
from pathlib import Path
import torch
from benchmark_cnn_resnet import measure_batch,file_hash
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pv_fault.models.baseline import SmallCNN
from pv_fault.models.resnet import ResNet18Thermal
from pv_fault.resnet_spatial import configure_spatial_variant


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2112)
    protocol=json.loads((ROOT/'configs/resnet18-spatial-v1.json').read_text())
    ledger=json.loads((ROOT/'results/resnet18-spatial-v1-ledger.json').read_text())
    if ledger['status']!='complete_verified': raise ValueError('Finish both fits and checkpoint checks first.')
    models=[]
    for name,run_id in [('standard',protocol['reference_run'])]+[(t['arm'],t['run_id']) for t in protocol['planned_trials']]:
        folder=ROOT/'runs'/run_id
        plan=json.loads((folder/'experiment-plan.json').read_text())
        verification=json.loads((folder/'checkpoint_verification.json').read_text())
        if not verification['checkpoint_predictions_verified'] or verification['checkpoint_sha256']!=file_hash(folder/'best_model.pt'):
            raise ValueError('Benchmark requires a verified unchanged checkpoint.')
        checkpoint=torch.load(folder/'best_model.pt',map_location='cpu',weights_only=True)
        variant=plan.get('spatial_variant','standard')
        if checkpoint.get('spatial_variant','standard')!=variant: raise ValueError('Variant metadata differs.')
        model=ResNet18Thermal(12,ROOT/plan['pretrained_weights'],plan['pretrained_sha256'])
        configure_spatial_variant(model,variant)
        model.load_state_dict(checkpoint['model_state_dict'])
        models.append((name,model,folder))
    folder=ROOT/'runs/smallcnn-depth-gentle60-s2112-v1'
    model=SmallCNN(12,batch_norm=True,extra_conv=True,activation='relu')
    model.load_state_dict(torch.load(folder/'best_model.pt',map_location='cpu',weights_only=True)['model_state_dict'])
    models.append(('SmallCNN',model,folder))
    inputs={size:torch.rand(size,1,40,24) for size in (1,64)}
    records=[]
    for name,model,folder in models:
        model.eval(); geometry={};handles=[]
        if name!='SmallCNN':
            def capture(stage):
                def hook(module,args,output): geometry[stage]=list(output.shape[1:])
                return hook
            for stage in ('conv1','maxpool','layer1','layer4','avgpool'):
                handles.append(getattr(model.backbone,stage).register_forward_hook(capture(stage)))
            with torch.inference_mode(): model(inputs[1])
            for handle in handles: handle.remove()
        record=dict(model=name,run_id=folder.name,geometry=geometry,
                    checkpoint_sha256=file_hash(folder/'best_model.pt'),
                    total_parameters=sum(p.numel() for p in model.parameters()),
                    parameter_storage_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                    batches=[measure_batch(model,inputs[size]) for size in (1,64)])
        records.append(record)
        print(name,record['batches'][0]['median_batch_ms'],'ms/image;',record['batches'][1]['median_batch_ms'],'ms/batch64',flush=True)
    result=dict(models=records,measured_utc=datetime.now(timezone.utc).isoformat(),cpu_threads=4,device='cpu',
                torch=torch.__version__,python=sys.version,platform=platform.platform(),test_evaluated=False,
                source_sha256={name:file_hash(ROOT/name) for name in ('scripts/benchmark_resnet_spatial.py','scripts/benchmark_cnn_resnet.py','src/pv_fault/resnet_spatial.py','src/pv_fault/models/resnet.py','src/pv_fault/models/baseline.py')},
                scope='Actual checkpoint weights, same synthetic grayscale inputs [N,1,40,24]. Five warmups then twenty timed model-only CPU forwards per batch size; after both fits finish. No dataset images or application/image-loading overhead. Parameter storage excludes buffers, gradients, optimiser state and activations; it is not peak RAM, FLOPs or energy.')
    (ROOT/'results/resnet18-spatial-v1-inference.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
