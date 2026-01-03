import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(Path(__file__).stem)


# ----------------------------
# Building blocks
# ----------------------------


class SpatialDownsample(nn.Module):
    """Downsample spatial features to a small grid.

    Preserves coarse spatial layout (e.g., food relative to head) better than
    global pooling, with minimal complexity.
    """

    def __init__(self, in_channels: int, out_hw: int = 4):
        super().__init__()
        self.out_hw = out_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C, out_hw, out_hw)
        x = F.adaptive_avg_pool2d(x, (self.out_hw, self.out_hw))
        return x

class SnakePPONet(nn.Module):
    """Compact PPO network with spatial downsample pooling.

    Inputs (dict):
    - 'map': (B,C,H,W)
    - 'ctx': (B,ctx_dim)
    - 'action_features': (B,A,F) with F=1 [margin_frac only]

        Design:
        - Spatial trunk: Conv(in_channels→64) → Conv(64→128) + residual blocks.
        - Pooling: AdaptiveAvgPool to a small grid (default 4x4), then flatten.
        - Policy head: dual-pathway (spatial + safety) mixed by a learnable gate.
        - Value head: pooled spatial features + context → scalar V(s).
    
    Key improvements:
    - Spatial attention preserves directional information (unlike GlobalAvgPool)
    - Model learns "what to look at" on the map
    - Better spatial reasoning with minimal parameter overhead (~500 extra params)
    """

    NUM_ACTIONS = 4
    ACTION_FEAT_DIM = 1  # Only margin_frac; removed safety_hint and food_hint shortcuts

    def __init__(self, in_channels: int, ctx_dim: int = 0):
        super().__init__()
        self.ctx_dim = ctx_dim

        # ----------------------------
        # Spatial trunk
        # ----------------------------
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Two simple residual blocks (each: 2x Conv3x3 with skip connection).
        # Inlined here for readability; mathematically equivalent to a small module.
        self.res1_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res2_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Before pooling, use a lightweight 1x1 bottleneck to keep the MLP heads small
        # while allowing a higher-resolution pooled grid.
        #
        # Rationale (walls / corridors): pooling too aggressively (e.g. to 4x4) tends to
        # smear thin walls and narrow passages. A higher pooled grid preserves geometry.
        self.prepool_bottleneck = nn.Conv2d(128, 64, kernel_size=1)

        # Compress to a small grid (default 8x8) instead of pooling to a single vector.
        # Keeps directional/spatial info needed for corridors and long-range food.
        self.spatial_pool = SpatialDownsample(64, out_hw=8)
        self._spatial_hw = 8

        # ----------------------------
        # Value head (critic)
        # ----------------------------
        value_input_dim = (64 * self._spatial_hw * self._spatial_hw) + ctx_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # ----------------------------
        # Policy head (actor)
        # Dual-pathway: spatial + safety pathways learn complementary signals.
        # ----------------------------
        base_dim = (64 * self._spatial_hw * self._spatial_hw) + ctx_dim
        self.base_proj = nn.Sequential(
            nn.Linear(base_dim, 128),
            nn.ReLU()
        )

        # Minimal action-conditioning for the spatial pathway.
        # Without this, spatial logits would be identical across actions.
        self.action_id_emb = nn.Embedding(self.NUM_ACTIONS, 8)
        
        # Spatial pathway: base embedding → per-action logits (food-seeking)
        self.spatial_action_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Safety pathway: action_features → per-action logits (safety preference)
        self.action_feat_proj = nn.Sequential(
            nn.Linear(self.ACTION_FEAT_DIM, 16),
            nn.ReLU()
        )
        self.safety_action_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    # Learnable importance weight for each pathway.
    # alpha = sigmoid(pathway_mixer): 0=all safety, 1=all spatial.
    # Initialize biased toward spatial: alpha ~= 0.7 -> pathway_mixer ~= logit(0.7) ~= 0.8473.
        self.pathway_mixer = nn.Parameter(torch.tensor(0.8473), requires_grad=True)


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

        # ----------------------------
        # Spatial trunk
        # ----------------------------
        # Ensure NHWC-friendly layout for better CUDA perf on convs
        if map_tensor.is_cuda:
            map_tensor = map_tensor.contiguous(memory_format=torch.channels_last)

        h = F.relu(self.conv1(map_tensor))
        h = F.relu(self.conv2(h))  # (B,128,H,W)

        # Residual block 1
        r = F.relu(self.res1_conv1(h))
        r = self.res1_conv2(r)
        h = F.relu(h + r)

        # Residual block 2
        r = F.relu(self.res2_conv1(h))
        r = self.res2_conv2(r)
        h = F.relu(h + r)

        h = F.relu(self.prepool_bottleneck(h))  # (B,64,H,W)

        h_small = self.spatial_pool(h)  # (B,64,8,8)
        pooled = h_small.reshape(h_small.size(0), -1)  # (B,64*8*8)
        base_input = torch.cat([pooled, ctx], dim=1)
        base_emb = self.base_proj(base_input)  # (B,128)

        # ----------------------------
        # Policy logits (dual-pathway)
        # ----------------------------
        # Spatial pathway: food-seeking from spatial map
        base_expanded = base_emb.unsqueeze(1).expand(B, A, base_emb.size(1))  # (B,A,128)
        action_ids = torch.arange(A, device=base_emb.device)
        action_emb = self.action_id_emb(action_ids).unsqueeze(0).expand(B, A, -1)  # (B,A,8)
        spatial_in = torch.cat([base_expanded, action_emb], dim=2)  # (B,A,136)
        spatial_flat = spatial_in.reshape(B * A, spatial_in.size(2))  # (B*A,136)
        spatial_logits_flat = self.spatial_action_head(spatial_flat)  # (B*A,1)
        spatial_logits = spatial_logits_flat.reshape(B, A)  # (B,A)

        # Safety pathway: preference from margin_frac
        safety_emb = self.action_feat_proj(action_features)  # (B,A,16)
        action_flat = safety_emb.reshape(B * A, safety_emb.size(2))  # (B*A,16)
        safety_logits_flat = self.safety_action_head(action_flat)  # (B*A,1)
        safety_logits = safety_logits_flat.reshape(B, A)  # (B,A)

        # Combine pathways with learned importance weight
        # alpha = sigmoid(pathway_mixer): 0=all safety, 1=all spatial
        alpha = torch.sigmoid(self.pathway_mixer)  # Learnable balance between 0 and 1
        action_logits = alpha * spatial_logits + (1 - alpha) * safety_logits  # (B,A)

        # ----------------------------
        # Value head
        # ----------------------------
        value_input = torch.cat([pooled, ctx], dim=1)
        values = self.value_head(value_input)  # (B,1)

        return action_logits, values


def model_factory(in_channels: int, ctx_dim: int) -> nn.Module:
    """Factory function to create PPO model instances."""
    log.info("Creating SnakePPONet model")
    return SnakePPONet(in_channels, ctx_dim)