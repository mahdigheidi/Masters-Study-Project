# A dead ReLU unit is one whose activation is always ≤ 0.



import torch


@torch.no_grad()
def compute_dead_units(model, dataloader, device="cuda"):
    model.eval()

    activations = {}

    def hook_fn(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    dead_counts = {}
    total_counts = {}

    for x, _ in dataloader:
        x = x.to(device)

        model(x)

        for name, act in activations.items():
            flat = act.view(act.size(0), act.size(1), -1)

            alive = (flat > 0).any(dim=(0, 2))

            if name not in dead_counts:
                dead_counts[name] = (~alive).float()
                total_counts[name] = torch.ones_like(alive).float()
            else:
                dead_counts[name] += (~alive).float()
                total_counts[name] += 1

    ratios = {}

    for name in dead_counts:
        ratios[name] = (
            dead_counts[name] / total_counts[name]
        ).mean().item()

    for h in hooks:
        h.remove()

    return ratios