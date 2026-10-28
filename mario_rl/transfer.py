"""Initialize a NEW experiment from an existing local pixel-policy checkpoint."""
import torch

from .envs import ACTION_SETS


def initialize_policy(model, checkpoint, actions, frame_skip):
    """Reuse the feature extractor and matching action rows; reset the critic.

    Button combinations, not row indices, identify actions. New actions keep
    their small random initialization with a conservative prior bias. Optimizer,
    learning-rate schedule, counters and recent episodes belong to the new run.
    """
    old_cfg = checkpoint['config']
    if old_cfg['frame_skip'] != frame_skip:
        raise ValueError('Warm start requires the same frame_skip/preprocessing')
    old_actions = ACTION_SETS[old_cfg['actions']]
    new_actions = ACTION_SETS[actions]
    old = checkpoint['model']
    target = model.state_dict()
    body = {k: v for k, v in old.items() if k.startswith('body.')}
    expected = {k for k in target if k.startswith('body.')}
    if set(body) != expected or any(body[k].shape != target[k].shape for k in expected):
        raise ValueError('Warm start feature extractor is incompatible')
    if old['actor.weight'].shape != (len(old_actions), model.actor.in_features):
        raise ValueError('Warm start actor is incompatible')
    if old['actor.bias'].shape != (len(old_actions),):
        raise ValueError('Warm start actor bias is incompatible')
    if not all(torch.isfinite(value).all() for value in body.values()):
        raise ValueError('Warm start contains nonfinite feature weights')
    if not torch.isfinite(old['actor.weight']).all() or not torch.isfinite(old['actor.bias']).all():
        raise ValueError('Warm start contains nonfinite actor weights')
    source_rows = {frozenset(buttons): i for i, buttons in enumerate(old_actions)}
    if not any(frozenset(buttons) in source_rows for buttons in new_actions):
        raise ValueError('No matching actions for warm start')
    model.load_state_dict(body, strict=False)
    copied = []
    with torch.no_grad():
        for i, buttons in enumerate(new_actions):
            j = source_rows.get(frozenset(buttons))
            if j is None:
                model.actor.bias[i].fill_(-4.0)
            else:
                model.actor.weight[i].copy_(old['actor.weight'][j])
                model.actor.bias[i].copy_(old['actor.bias'][j])
                copied.append(list(buttons))
    return {'source_step': checkpoint.get('global_step'), 'copied_actions': copied,
            'critic_reset': True, 'optimizer_reset': True}
