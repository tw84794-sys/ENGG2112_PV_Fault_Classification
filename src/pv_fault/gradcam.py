"""Class-specific Grad-CAM without changing weights or accumulating parameter gradients."""
import torch


def grad_cam(model, images, target_layer, class_indices):
    """Return logits and per-image [0, 1] maps on the layer's native spatial grid."""
    if model.training:
        raise ValueError("Grad-CAM requires an evaluation-mode model.")
    if images.ndim != 4 or not images.shape[0] or not torch.isfinite(images).all():
        raise ValueError("Expected a finite, nonempty image batch [N, C, H, W].")
    features = []
    hook = target_layer.register_forward_hook(lambda module, inputs, output: features.append(output))
    try:
        with torch.enable_grad():
            logits = model(images.detach().clone().requires_grad_(True))
            targets = torch.as_tensor(class_indices, device=logits.device)
            if (targets.dtype != torch.long or targets.shape != (images.shape[0],)
                    or (targets < 0).any() or (targets >= logits.shape[1]).any()):
                raise ValueError("Choose one valid integer class index per image.")
            if len(features) != 1 or features[0].ndim != 4:
                raise ValueError("Target layer must produce one spatial feature tensor.")
            activations = features[0]
            gradients, = torch.autograd.grad(logits.gather(1, targets[:, None]).sum(), activations)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            maps = (weights * activations).sum(dim=1, keepdim=True).relu()
            peaks = maps.amax(dim=(2, 3), keepdim=True)
            maps = maps / peaks.clamp_min(torch.finfo(maps.dtype).eps)
            return logits.detach(), maps.detach()
    finally:
        hook.remove()
