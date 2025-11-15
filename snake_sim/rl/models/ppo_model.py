import logging 
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Union, Dict, Tuple

log = logging.getLogger(Path(__file__).stem)

class SnakePPONet(nn.Module):
    """Simple stable PPO network for snake environment."""
    
    def __init__(self, in_channels: int, ctx_dim: int = 0):
        super().__init__()
        
        # Simple CNN feature extraction
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Policy head (actor)
        policy_input_dim = 64 + ctx_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 actions: up, down, left, right
        )
        
        # Value head (critic)  
        self.value_head = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
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
        
        # Global pooling to get fixed-size features
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1)  # (B, 64)
        
        # Concatenate context if available
        if ctx is not None:
            features_with_ctx = torch.cat([pooled, ctx], dim=1)
        else:
            features_with_ctx = pooled
            
        # Get policy and value outputs
        action_logits = self.policy_head(features_with_ctx)
        values = self.value_head(features_with_ctx)
        
        return action_logits, values


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


def model_factory(in_channels: int, ctx_dim: int = 0, map_size: int = 32, advanced: bool = False) -> nn.Module:
    """Factory function to create PPO model instances."""
    if advanced:
        return SpatialSnakePPONet(in_channels, ctx_dim, map_size)
    else:
        return SnakePPONet(in_channels, ctx_dim)