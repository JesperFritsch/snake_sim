"""SnakePPONet — conv frontend + transformer trunk + head-anchored policy head.

Architecture overview
---------------------
1. Conv frontend (2 layers): cheap local feature extraction.
2. Transformer encoder (2 layers, pre-norm): every cell can attend to every
   other cell, giving true global context regardless of board distance.
3. Head-anchored policy head: per-action logits are produced from
   (head_feat, action_cell_feat, scalar_action_features). The action_cell_feat
   is gathered at (head + action_offset) AFTER the attention layers, so its
   activation already encodes whole-board context — including geometry on the
   opposite side of the head.
4. Value head: mean over all tokens + ctx → V(s).

Inputs (dict): same contract as the previous model.
- 'map':              (B, C, H, W)
- 'ctx':              (B, ctx_dim)
- 'action_features':  (B, A, F)   F = ACTION_FEAT_DIM (currently 1: margin_frac)

Outputs:
- action_logits:      (B, A)
- values:             (B, 1)
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(Path(__file__).stem)


def _build_2d_sincos_pos_embedding(h: int, w: int, dim: int,
                                    device: torch.device,
                                    dtype: torch.dtype) -> torch.Tensor:
    """Standard 2D sinusoidal positional encoding, shape (h*w, dim).

    First half of `dim` encodes y, second half encodes x. Computed in fp32 then
    cast to the requested dtype to avoid precision artefacts in fp16 training.
    """
    assert dim % 4 == 0, f"pos embedding dim ({dim}) must be divisible by 4"
    half = dim // 2

    def sincos_1d(length: int, d: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        i = torch.arange(d // 2, dtype=torch.float32, device=device).unsqueeze(0)
        denom = torch.pow(torch.tensor(10000.0, device=device), (2 * i) / d)
        angles = pos / denom
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    y_emb = sincos_1d(h, half)                          # (h, half)
    x_emb = sincos_1d(w, half)                          # (w, half)
    grid_y = y_emb.unsqueeze(1).expand(h, w, half)      # (h, w, half)
    grid_x = x_emb.unsqueeze(0).expand(h, w, half)      # (h, w, half)
    pos = torch.cat([grid_y, grid_x], dim=2).reshape(h * w, dim)
    return pos.to(dtype=dtype)


class SnakePPONet(nn.Module):
    """PPO network with attention-based global reasoning and head-anchored policy.

    Compatible drop-in replacement for the previous SnakePPONet: same input dict
    keys, same NUM_ACTIONS / ACTION_FEAT_DIM constants used by the trainer.
    """

    NUM_ACTIONS = 4
    ACTION_FEAT_DIM = 1  # margin_frac. Bump this if you add Voronoi deltas etc.

    # Architecture hyperparameters
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_TRANSFORMER_LAYERS = 2
    FFN_MULT = 2

    # Channel index of the own-head one-hot in the input map.
    # Matches BaseStateBuilder._default_order: ['head', 'body', 'food', ...].
    HEAD_CHANNEL = 0

    # Per-action (dy, dx) offsets. MUST match constants.ACTION_ORDER:
    #   0: Up    Coord(0, -1) -> (dy=-1, dx= 0)
    #   1: Right Coord(1,  0) -> (dy= 0, dx= 1)
    #   2: Down  Coord(0,  1) -> (dy= 1, dx= 0)
    #   3: Left  Coord(-1, 0) -> (dy= 0, dx=-1)
    ACTION_OFFSETS: Tuple[Tuple[int, int], ...] = (
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    )

    def __init__(self, in_channels: int, ctx_dim: int = 0):
        super().__init__()
        self.ctx_dim = ctx_dim

        # ---- Conv frontend ----
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, self.EMBED_DIM, kernel_size=3, padding=1)

        # ---- Transformer trunk ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.EMBED_DIM,
            nhead=self.NUM_HEADS,
            dim_feedforward=self.EMBED_DIM * self.FFN_MULT,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-norm: more stable for training-from-scratch
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.NUM_TRANSFORMER_LAYERS,
        )

        # ---- Policy head ----
        # Per-action input: [head_feat (E), action_cell_feat (E), action_features (F)]
        policy_in = self.EMBED_DIM * 2 + self.ACTION_FEAT_DIM
        self.policy_head = nn.Sequential(
            nn.Linear(policy_in, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # ---- Value head ----
        # Mean-pool global summary + ctx -> scalar value.
        value_in = self.EMBED_DIM + ctx_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_in, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_head_pos(self, map_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recover (head_y, head_x) per-batch by argmax over the head channel.

        The head channel is a one-hot in BaseStateBuilder, so argmax is exact.
        """
        head_channel = map_tensor[:, self.HEAD_CHANNEL]   # (B, H, W)
        B, H, W = head_channel.shape
        flat_idx = head_channel.reshape(B, -1).argmax(dim=1)  # (B,)
        head_y = flat_idx // W                                 # (B,)
        head_x = flat_idx % W                                  # (B,)
        return head_y, head_x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            for k in ('map', 'ctx', 'action_features'):
                if k not in x:
                    raise KeyError(f"Missing required input key: {k}")

        map_tensor: torch.Tensor = x['map']               # (B, C, H, W)
        ctx: torch.Tensor = x['ctx']                       # (B, ctx_dim)
        action_features: torch.Tensor = x['action_features']  # (B, A, F)

        B, _C, H, W = map_tensor.shape
        A = self.NUM_ACTIONS

        if self.training:
            if action_features.shape != (B, A, self.ACTION_FEAT_DIM):
                raise ValueError(
                    f"action_features shape {tuple(action_features.shape)} "
                    f"!= expected (B={B}, A={A}, F={self.ACTION_FEAT_DIM})"
                )
            if ctx.shape != (B, self.ctx_dim):
                raise ValueError(
                    f"ctx shape {tuple(ctx.shape)} != expected (B={B}, {self.ctx_dim})"
                )

        # ---- Conv frontend ----
        h = F.relu(self.conv1(map_tensor))
        h = F.relu(self.conv2(h))                    # (B, E, H, W)

        # ---- Tokenize + positional encoding + transformer ----
        tokens = h.flatten(2).transpose(1, 2)        # (B, H*W, E)
        pos = _build_2d_sincos_pos_embedding(
            H, W, self.EMBED_DIM, tokens.device, tokens.dtype
        )                                            # (H*W, E)
        tokens = tokens + pos.unsqueeze(0)
        tokens = self.transformer(tokens)            # (B, H*W, E)

        # ---- Reshape back to spatial; pad by 1 so head-adjacent gathers are safe ----
        feat = tokens.transpose(1, 2).reshape(B, self.EMBED_DIM, H, W)
        feat_padded = F.pad(feat, (1, 1, 1, 1))      # (B, E, H+2, W+2), zero pad

        # ---- Head-anchored gather ----
        head_y, head_x = self._extract_head_pos(map_tensor)
        head_y_p = head_y + 1
        head_x_p = head_x + 1
        batch_idx = torch.arange(B, device=feat.device)

        # head_feat: (B, E)
        head_feat = feat_padded[batch_idx, :, head_y_p, head_x_p]

        # action_cell_feats: (B, A, E)
        action_cell_feats_list = []
        for (dy, dx) in self.ACTION_OFFSETS:
            ay = head_y_p + dy
            ax = head_x_p + dx
            action_cell_feats_list.append(feat_padded[batch_idx, :, ay, ax])
        action_cell_feats = torch.stack(action_cell_feats_list, dim=1)

        # ---- Policy head ----
        head_feat_expanded = head_feat.unsqueeze(1).expand(B, A, self.EMBED_DIM)
        policy_input = torch.cat(
            [head_feat_expanded, action_cell_feats, action_features], dim=2
        )                                                # (B, A, 2E + F)
        policy_input_flat = policy_input.reshape(B * A, -1)
        action_logits = self.policy_head(policy_input_flat).reshape(B, A)

        # ---- Value head ----
        global_summary = tokens.mean(dim=1)              # (B, E)
        value_input = torch.cat([global_summary, ctx], dim=1)
        values = self.value_head(value_input)            # (B, 1)

        return action_logits, values


def model_factory(in_channels: int, ctx_dim: int) -> nn.Module:
    """Factory used by ppo_trainer / ppo_snake. Signature unchanged."""
    log.info("Creating SnakePPONet (transformer) model")
    return SnakePPONet(in_channels, ctx_dim)