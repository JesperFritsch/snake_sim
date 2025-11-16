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


class SpatialSnakePPONet(nn.Module):
    """Advanced spatial-aware PPO network that maintains full spatial reasoning."""
    
    def __init__(self, in_channels: int, ctx_dim: int = 0, map_size: int = 32):
        super().__init__()
        self.map_size = map_size
        
        # Enhanced CNN with more spatial layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1)  # Reduce channels for final processing
        
        # Positional encoding to help with spatial reasoning
        self.register_buffer('pos_encoding', self._create_position_encoding(map_size))
        
        # Enhanced spatial policy head
        self.spatial_policy = nn.Sequential(
            nn.Conv2d(32 + 2, 16, kernel_size=3, padding=1),  # +2 for position encoding
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1)  # 4 actions
        )
        
        # Multi-scale value head
        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_pool_fine = nn.AdaptiveAvgPool2d(8)    # Fine spatial context
        self.value_pool_coarse = nn.AdaptiveAvgPool2d(2)  # Coarse spatial context
        
        value_input_dim = 32 * 8 * 8 + 32 * 2 * 2 + ctx_dim  # Multi-scale + context
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def _create_position_encoding(self, size: int) -> torch.Tensor:
        """Create normalized position encoding for spatial awareness."""
        y_pos = torch.arange(size).float().unsqueeze(1).repeat(1, size) / (size - 1)
        x_pos = torch.arange(size).float().unsqueeze(0).repeat(size, 1) / (size - 1)
        pos_encoding = torch.stack([y_pos, x_pos], dim=0)  # (2, H, W)
        return pos_encoding
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with enhanced spatial reasoning.
        
        Args:
            x: Either a tensor of shape (B, C, H, W) or dict with 'map' and optional 'ctx' keys
            
        Returns:
            Tuple of (action_logits, values)
        """
        if isinstance(x, dict):
            map_tensor = x['map']
            ctx = x.get('ctx', None)
        else:
            map_tensor = x
            ctx = None
            
        batch_size = map_tensor.shape[0]
            
        # Enhanced CNN feature extraction
        features = F.relu(self.conv1(map_tensor))
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features))
        
        # Store features for value computation before final reduction
        value_features = features
        
        # Reduce channels for policy computation
        policy_features = F.relu(self.conv4(features))  # (B, 32, H, W)
        
        # Add positional encoding
        pos_enc = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)
        policy_features_with_pos = torch.cat([policy_features, pos_enc], dim=1)
        
        # Spatial policy computation with position awareness
        spatial_logits = self.spatial_policy(policy_features_with_pos)  # (B, 4, H, W)
        
        # Intelligent spatial aggregation - weight by attention to important regions
        attention_weights = torch.softmax(spatial_logits.sum(dim=1, keepdim=True), dim=-1)  # (B, 1, H, W)
        attention_weights = attention_weights.view(batch_size, 1, -1)  # (B, 1, H*W)
        
        spatial_logits_flat = spatial_logits.view(batch_size, 4, -1)  # (B, 4, H*W)
        action_logits = torch.sum(spatial_logits_flat * attention_weights, dim=2)  # (B, 4)
        
        # Multi-scale value computation
        value_conv_features = self.value_conv(value_features)
        
        # Extract multi-scale features
        fine_features = self.value_pool_fine(value_conv_features).flatten(1)
        coarse_features = self.value_pool_coarse(value_conv_features).flatten(1)
        
        # Combine multi-scale features
        combined_features = torch.cat([fine_features, coarse_features], dim=1)
        
        # Add context if available
        if ctx is not None:
            combined_features = torch.cat([combined_features, ctx], dim=1)
            
        values = self.value_head(combined_features)
        
        return action_logits, values

    # ---- Parameter grouping helpers ----
    def shared_parameters(self):
        return itertools.chain(
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.conv3.parameters(),
            self.conv4.parameters(),
            self.value_conv.parameters()
        )

    def actor_parameters(self):
        return itertools.chain(self.shared_parameters(), self.spatial_policy.parameters())

    def critic_parameters(self):
        return self.value_head.parameters()


def model_factory(in_channels: int, ctx_dim: int = 0, map_size: int = 32, advanced: bool = False) -> nn.Module:
    """Factory function to create PPO model instances."""
    if advanced:
        log.info("Creating SpatialSnakePPONet model")
        return SpatialSnakePPONet(in_channels, ctx_dim, map_size)
    else:
        log.info("Creating SnakePPONet model")
        return SnakePPONet(in_channels, ctx_dim)