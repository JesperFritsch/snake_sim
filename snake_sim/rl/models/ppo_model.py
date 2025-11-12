import torch

from snake_sim.rl.constants import ACTION_ORDER

class SnakePPONet(torch.nn.Module):
    """Inline minimal CNN+MLP policy/value network for PPO.

    Inputs (single or batched):
      - map: torch.Tensor shape (B,C,H,W)
      - ctx (optional): torch.Tensor shape (B,K)

    Outputs:
      - logits: (B, A) unnormalized action scores
      - value:  (B, 1) state-value estimates
    """
    def __init__(self, in_channels: int, ctx_dim: int = 0):
        super().__init__()
        self.ctx_dim = ctx_dim
        # Lightweight conv stack
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self._feature_dim = 64  # channels after conv (global avg pool reduces to this)
        # Heads
        mlp_input = self._feature_dim + (ctx_dim if ctx_dim > 0 else 0)
        hidden = 128
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(mlp_input, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, len(ACTION_ORDER))
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(mlp_input, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # Accept either raw map tensor or dict {'map':..., 'ctx':...}
        if isinstance(x, dict):
            map_tensor = x['map']
            ctx = x.get('ctx')
        else:
            map_tensor = x
            ctx = None
        feats = self.conv(map_tensor)
        # Global average pool to (B, C)
        feats = feats.mean(dim=(-1, -2))  # (B,64)
        if ctx is not None:
            feats = torch.cat([feats, ctx], dim=-1)
        logits = self.policy_head(feats)
        value = self.value_head(feats)
        return logits, value
