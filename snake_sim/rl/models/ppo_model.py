import logging 
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from pathlib import Path
from typing import Union, Dict, Tuple

log = logging.getLogger(Path(__file__).stem)

class SnakePPONet(nn.Module):
    """Simplified PPO network assuming consistent input dict with keys:
    'map': (B,C,H,W), 'ctx': (B,ctx_dim), 'action_features': (B,A,F).

    Removes all fallback / branching logic: training code must supply all tensors.
    Policy logits are produced via per-action conditioning (spatial trunk + ctx + per-action feats).
    Value head ignores per-action features (scalar V(s)).
    """

    NUM_ACTIONS = 4
    ACTION_FEAT_DIM = 4  # [margin_frac, safety_hint, food_hint, valid_mask]

    def __init__(self, in_channels: int, ctx_dim: int = 0):
        super().__init__()
        self.ctx_dim = ctx_dim

        # Spatial trunk
        self.conv1 = nn.Conv2d(in_channels,64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.spatial_reduce = nn.AdaptiveAvgPool2d(16)  # -> (64,16,16)
        trunk_flat_dim = 64 * 16 * 16  # 16384

        # Value head (critic)
        value_input_dim =64 + ctx_dim
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Policy per-action conditioning
        base_input_dim = trunk_flat_dim + ctx_dim  # spatial flat + ctx
        self.base_proj = nn.Linear(base_input_dim, 128)
        self.action_feat_proj = nn.Linear(self.ACTION_FEAT_DIM, 32)
        self.per_action_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128 + 32, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    # NOTE: Single-optimizer design; no actor/critic parameter grouping needed.

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Enforce presence of required keys for clarity
        if self.training:
            # During training, validate inputs explicitly.
            if not all(k in x for k in ('map', 'ctx', 'action_features')):
                missing = [k for k in ('map', 'ctx', 'action_features') if k not in x]
                raise KeyError(f"Missing required input keys: {missing}")
        map_tensor = x['map']
        ctx = x['ctx']
        action_features = x['action_features']  # (B,A,F)

        # Guard shape checks only in training to avoid TorchScript tracer warnings.
        if self.training:
            B, A, Fdim = action_features.shape
            if A != self.NUM_ACTIONS or Fdim != self.ACTION_FEAT_DIM:
                raise ValueError(f"Unexpected action_features shape {action_features.shape}; expected (B,{self.NUM_ACTIONS},{self.ACTION_FEAT_DIM})")
            if ctx.shape[1] != self.ctx_dim:
                raise ValueError(f"ctx dim mismatch: got {ctx.shape[1]} expected {self.ctx_dim}")
        else:
            # Infer B for downstream reshapes (TorchScript tracing path) without Python conditionals.
            B = action_features.shape[0]
            A = self.NUM_ACTIONS

        # Spatial trunk
        h = F.relu(self.conv1(map_tensor))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))  # (B,64,H,W)

        # Policy trunk flatten
        policy_spatial = self.spatial_reduce(h)  # (B,64,16,16)
        trunk_flat = policy_spatial.flatten(1)  # (B,16384)
        base_input = torch.cat([trunk_flat, ctx], dim=1)  # (B,16384+ctx_dim)
        base_emb = self.base_proj(base_input)  # (B,128)

        # Per-action conditioning
        action_emb = self.action_feat_proj(action_features)  # (B,A,32)
        base_expanded = base_emb.unsqueeze(1).expand(B, A, base_emb.size(1))  # (B,A,128)
        combined = torch.cat([base_expanded, action_emb], dim=2)  # (B,A,160)
        combined_flat = combined.view(B * A, combined.size(2))  # (B*A,160)
        logits_flat = self.per_action_head(combined_flat)  # (B*A,1)
        action_logits = logits_flat.view(B, A)  # (B,A)

        # Value head
        value_spatial = self.value_pool(h).squeeze(-1).squeeze(-1)  # (B,64)
        value_input = torch.cat([value_spatial, ctx], dim=1)
        values = self.value_head(value_input)  # (B,1)

        return action_logits, values


def model_factory(in_channels: int, ctx_dim: int) -> nn.Module:
    """Factory function to create PPO model instances."""
    log.info("Creating SnakePPONet model")
    return SnakePPONet(in_channels, ctx_dim)