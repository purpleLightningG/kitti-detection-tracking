"""
weight_mapper.py — Load OpenPCDet PointPillars checkpoint into the standalone model.

Now that the standalone module matches OpenPCDet's exact layer indexing
(ZeroPad → Conv → BN → ReLU pattern, same Sequential numbering), the only
remapping needed is:
    vfe              → pillar_feature_net
    pfn_layers.X.linear  → pfn_layers.X.conv  (Linear vs Conv1d)
    dense_head       → detection_head

Plus reshape: Linear weight (out, in) → Conv1d weight (out, in, 1).
"""
import torch
from collections import OrderedDict


def remap_pcdet_to_standalone(pcdet_state_dict: OrderedDict):
    """Convert OpenPCDet PointPillars state_dict to standalone naming."""
    new_state = OrderedDict()
    unmatched = []

    for key, val in pcdet_state_dict.items():
        new_key = _rename_key(key)
        if new_key is None:
            unmatched.append(key)
            continue

        # Linear (out, in) → Conv1d (out, in, 1) for PFN layers
        if 'pillar_feature_net.pfn_layers' in new_key and new_key.endswith('.conv.weight'):
            if val.dim() == 2:
                val = val.unsqueeze(-1)

        new_state[new_key] = val

    return new_state, unmatched


def _rename_key(pcdet_key: str):
    # Skip global_step counter
    if pcdet_key == 'global_step':
        return None

    # VFE → pillar_feature_net, linear → conv
    if pcdet_key.startswith('vfe.'):
        new_key = pcdet_key.replace('vfe.', 'pillar_feature_net.', 1)
        new_key = new_key.replace('.linear.', '.conv.')
        return new_key

    # Backbone keeps identical layer indices
    if pcdet_key.startswith('backbone_2d.'):
        return pcdet_key

    # Detection head: dense_head → detection_head
    if pcdet_key.startswith('dense_head.'):
        return pcdet_key.replace('dense_head.', 'detection_head.', 1)

    return None


def load_pcdet_checkpoint_into_standalone(
    checkpoint_path: str,
    standalone_model,
    verbose: bool = True,
):
    """Load and remap weights from OpenPCDet checkpoint into a standalone model."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pcdet_state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))

    new_state, unmatched = remap_pcdet_to_standalone(pcdet_state)

    if verbose:
        print(f"[WeightMapper] Loaded {len(pcdet_state)} keys from checkpoint")
        print(f"[WeightMapper] Mapped {len(new_state)} keys to standalone naming")
        if unmatched:
            print(f"[WeightMapper] Skipped {len(unmatched)} non-trainable keys: {unmatched[:3]}")

    missing, unexpected = standalone_model.load_state_dict(new_state, strict=False)

    if verbose:
        if missing:
            print(f"[WeightMapper] WARNING: {len(missing)} keys MISSING in standalone model:")
            for k in missing[:10]:
                print(f"    {k}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
        if unexpected:
            print(f"[WeightMapper] WARNING: {len(unexpected)} UNEXPECTED keys in checkpoint:")
            for k in unexpected[:10]:
                print(f"    {k}")
            if len(unexpected) > 10:
                print(f"    ... and {len(unexpected) - 10} more")
        if not missing and not unexpected:
            print(f"[WeightMapper] ✓ All weights loaded successfully!")

    return missing, unexpected
