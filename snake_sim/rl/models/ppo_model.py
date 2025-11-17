import logging 
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from pathlib import Path
from typing import Union, Dict, Tuple

log = logging.getLogger(Path(__file__).stem)

class SnakePPONet(nn.Module):
    """Simple stable PPO network for snake environment - NOW WITH SPATIAL REASONING."""
    
    def __init__(self, in_channels: int, ctx_dim: int = 0):
        super().__init__()
        
        # CNN feature extraction - keep spatial info
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Spatial policy head - predicts action logits at head position
        # We'll pool to 4x4 to keep SOME spatial awareness without being huge
        self.spatial_reduce = nn.AdaptiveAvgPool2d(4)  # 32x32 -> 4x4 (reduces params)
        
        # Policy head (actor) - uses 4x4 spatial features
        policy_input_dim = 32 * 4 * 4 + ctx_dim  # 512 + ctx_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 actions: up, down, left, right
        )
        
        # Value head (critic) - can use global pooling since it's just a scalar
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        value_input_dim = 32 + ctx_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Either a tensor of shape (B, C, H, W) or dict with 'map' and optional 'ctx' keys
            
        Returns:
            Tuple of (action_logits, values) where:
            - action_logits: (B, 4) - logits for each action
            - values: (B, 1) - state value estimates
        """
        if isinstance(x, dict):
            map_tensor = x['map']
            ctx = x.get('ctx', None)
        else:
            map_tensor = x
            ctx = None
            
        # CNN feature extraction
        features = F.relu(self.conv1(map_tensor))
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features))  # (B, 32, H, W)
        
        # Policy uses spatial features (4x4)
        policy_features = self.spatial_reduce(features)  # (B, 32, 4, 4)
        policy_flat = policy_features.flatten(1)  # (B, 512)
        
        # Value uses global pooling (just needs scalar)
        value_features = self.value_pool(features).squeeze(-1).squeeze(-1)  # (B, 32)
        
        # Concatenate context if available
        if ctx is not None:
            policy_input = torch.cat([policy_flat, ctx], dim=1)
            value_input = torch.cat([value_features, ctx], dim=1)
        else:
            policy_input = policy_flat
            value_input = value_features
            
        # Get policy and value outputs
        action_logits = self.policy_head(policy_input)
        values = self.value_head(value_input)
        
        return action_logits, values

    # ---- Parameter grouping helpers ----
    def shared_parameters(self):
        """Convolutional trunk shared by actor and critic."""
        return itertools.chain(
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.conv3.parameters()
        )

    def actor_parameters(self):
        """Actor params: shared trunk + policy head."""
        return itertools.chain(self.shared_parameters(), self.policy_head.parameters())

    def critic_parameters(self):
        """Critic params: value head only to avoid double optimizer stepping of trunk."""
        return self.value_head.parameters()



def model_factory(in_channels: int, ctx_dim: int = 0) -> nn.Module:
    """Factory function to create PPO model instances."""
    log.info("Creating SnakePPONet model")
    return SnakePPONet(in_channels, ctx_dim)