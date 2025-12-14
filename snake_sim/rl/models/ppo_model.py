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

    def __init__(self, in_channels: int, ctx_dim: int = 0, af_dropout_prob: float = 0.2):
        super().__init__()
        self.ctx_dim = ctx_dim
        # Probability to drop entire per-action feature vectors during training
        # (applied per-action, per-sample). Default 0.0 (no dropout).
        self.af_dropout_prob = float(af_dropout_prob)

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
        # Reduce AF capacity and add gate to prevent over-reliance early
        self.action_feat_proj = nn.Linear(self.ACTION_FEAT_DIM, 16)
        # Learnable gate (scalar) on action-feature path, initialized to bias closed
        self.af_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ~ 0.12
        # Directional per-action map embedding (from last conv feature map)
        # We will pool features along a short ray in each action direction
        self.dir_proj = nn.Linear(64, 32)
        self.per_action_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128 + 32 + 16, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


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
        # Optionally apply per-action-feature dropout during training to
        # discourage over-reliance on action_features.
        if self.training and self.af_dropout_prob > 0.0:
            keep_prob = 1.0 - self.af_dropout_prob
            # action_features shape: (B,A,F)
            mask = torch.rand(action_features.shape[:2], device=action_features.device) < keep_prob
            mask = mask.unsqueeze(-1).to(action_features.dtype)
            action_features = action_features * mask

        # Compute simple directional map embeddings per action by averaging
        # features k steps ahead from the head position along each direction.
        # We don't have head coords here, so approximate by pooling from the
        # central region with directional weighting. This is cheap and avoids
        # plumbing extra inputs.
        B = map_tensor.shape[0]
        _, C, H, W = h.shape
        # Create fixed directional weight masks over the 16x16 reduced map
        # up, right, down, left roughly emphasize forward half.
        with torch.no_grad():
            yy, xx = torch.meshgrid(torch.linspace(-1,1,16, device=h.device),
                                     torch.linspace(-1,1,16, device=h.device), indexing='ij')
            up_mask = (yy < 0).float()
            down_mask = (yy > 0).float()
            right_mask = (xx > 0).float()
            left_mask = (xx < 0).float()
            dir_masks = torch.stack([up_mask, right_mask, down_mask, left_mask], dim=0)  # (A,16,16)
            dir_masks = dir_masks / (dir_masks.sum(dim=(1,2), keepdim=True) + 1e-6)
        # Apply masks to reduced spatial features
        # policy_spatial: (B,64,16,16)
        ps = policy_spatial  # alias
        # Compute masked average per action
        ps_flat = ps.view(B, C, 16*16)
        masks_flat = dir_masks.view(self.NUM_ACTIONS, 16*16).unsqueeze(0).expand(B, -1, -1)  # (B,A,256)
        # Weighted sum over spatial positions
        dir_feats = torch.einsum('bci,bai->bac', ps_flat, masks_flat)  # (B,A,64)
        dir_emb = self.dir_proj(dir_feats)  # (B,A,32)

        action_emb = self.action_feat_proj(action_features)  # (B,A,16)
        # Gate the action-feature path
        af_gate = torch.sigmoid(self.af_gate)  # scalar in (0,1)
        action_emb = action_emb * af_gate
        base_expanded = base_emb.unsqueeze(1).expand(B, A, base_emb.size(1))  # (B,A,128)
        combined = torch.cat([base_expanded, dir_emb, action_emb], dim=2)  # (B,A,128+32+16)
        combined_flat = combined.view(B * A, combined.size(2))  # (B*A,176)
        logits_flat = self.per_action_head(combined_flat)  # (B*A,1)
        action_logits = logits_flat.view(B, A)  # (B,A)

        # Value head
        value_spatial = self.value_pool(h).squeeze(-1).squeeze(-1)  # (B,64)
        value_input = torch.cat([value_spatial, ctx], dim=1)
        values = self.value_head(value_input)  # (B,1)

        return action_logits, values


def model_factory(in_channels: int, ctx_dim: int, af_dropout_prob: float = 0.2) -> nn.Module:
    """Factory function to create PPO model instances."""
    log.info("Creating SnakePPONet model")
    return SnakePPONet(in_channels, ctx_dim, af_dropout_prob)