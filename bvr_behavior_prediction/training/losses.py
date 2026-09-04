def multitask_cross_entropy(outputs, targets, weights=None):
    import torch.nn.functional as functional
    weights = weights or {}; losses = {}
    for name, logits in outputs.items():
        if name in targets: losses[name] = functional.cross_entropy(logits, targets[name])
    if not losses: raise ValueError("No output names match target names")
    return sum(weights.get(name, 1.0) * loss for name, loss in losses.items()), losses

