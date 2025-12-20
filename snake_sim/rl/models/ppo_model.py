import logging 
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from pathlib import Path
from typing import Union, Dict, Tuple

log = logging.getLogger(Path(__file__).stem)

class SnakePPONet(nn.Module):
    """Compact PPO network.

    Inputs (dict):
    - 'map': (B,C,H,W)
    - 'ctx': (B,ctx_dim)
    - 'action_features': (B,A,F) with F=3 [margin_frac, safety_hint, food_hint]

    Design:
    - Small conv trunk (C→32→64) with global average pooling to 64-d state embedding.
    - Policy head: per-action conditioning via concatenation of state embedding and action feature embedding.
    - Value head: pooled map + ctx → scalar.
    """

    NUM_ACTIONS = 4
    ACTION_FEAT_DIM = 3  # [margin_frac, safety_hint, food_hint]

    def __init__(self, in_channels: int, ctx_dim: int = 0, af_dropout_prob: float = 0.1):
        super().__init__()
        self.ctx_dim = ctx_dim
        # Probability to drop entire per-action feature vectors during training
        self.af_dropout_prob = float(af_dropout_prob)

        # Spatial trunk (wider, still simple)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (128,1,1)

        # Value head (critic)
        value_input_dim = 128 + ctx_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Policy per-action conditioning (compact)
        base_dim = 128 + ctx_dim
        # Add nonlinearity after projection to increase expressivity
        self.base_proj = nn.Sequential(
            nn.Linear(base_dim, 128),
            nn.ReLU()
        )
        # We'll concat an action-index one-hot to action_features at forward time
        self.register_buffer('action_one_hot', torch.eye(self.NUM_ACTIONS))
        self.action_feat_proj = nn.Sequential(
            nn.Linear(self.ACTION_FEAT_DIM + self.NUM_ACTIONS, 32),
            nn.ReLU()
        )
        self.per_action_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
        # Ensure NHWC-friendly layout for better CUDA perf on convs
        if map_tensor.is_cuda:
            map_tensor = map_tensor.contiguous(memory_format=torch.channels_last)

        h = F.relu(self.conv1(map_tensor))
        h = F.relu(self.conv2(h))  # (B,128,H,W)
        pooled = self.global_pool(h).squeeze(-1).squeeze(-1)  # (B,128)
        base_input = torch.cat([pooled, ctx], dim=1)  # (B,128+ctx_dim)
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

        # Concatenate a fixed action-index one-hot vector so the network has
        # an explicit positional encoding of each action. This makes the model
        # robust to permutations of action_features and improves learnability.
        B = action_features.shape[0]
        A = action_features.shape[1]
        oh = self.action_one_hot.unsqueeze(0).expand(B, A, -1).to(action_features.device)
        # Ensure action_features is a floating tensor
        if not torch.is_floating_point(action_features):
            action_features = action_features.float()
        action_features = torch.cat([action_features, oh], dim=2)

        # Per-action conditioning
        action_emb = self.action_feat_proj(action_features)  # (B,A,16)
        base_expanded = base_emb.unsqueeze(1).expand(B, A, base_emb.size(1))  # (B,A,128)
        combined = torch.cat([base_expanded, action_emb], dim=2)  # (B,A,144)
        combined_flat = combined.view(B * A, combined.size(2))  # (B*A,144)
        logits_flat = self.per_action_head(combined_flat)  # (B*A,1)
        action_logits = logits_flat.view(B, A)  # (B,A)

        # Value head
        value_input = torch.cat([pooled, ctx], dim=1)
        values = self.value_head(value_input)  # (B,1)

        return action_logits, values


def model_factory(in_channels: int, ctx_dim: int, af_dropout_prob: float = 0.0) -> nn.Module:
    """Factory function to create PPO model instances."""
    log.info("Creating SnakePPONet model")
    return SnakePPONet(in_channels, ctx_dim, af_dropout_prob)