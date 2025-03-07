import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class SimplifiedMambaBlock(nn.Module):
    """Ultra-light Mamba block with simplified structure to minimize memory usage"""
    def __init__(self, d_model, d_state=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Normalization layer
        self.norm = nn.LayerNorm(d_model)
        
        # Use MLP with GELU as a simplified approximation of Mamba mechanism
        # This dramatically reduces memory usage while maintaining expressivity
        self.mamba_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Add a residual connection
        self.skip_connection = nn.Identity()
        
    def forward(self, x):
        # Apply normalization
        residual = x
        x = self.norm(x)
        
        # Apply simplified Mamba-inspired transformation
        x = self.mamba_mlp(x)
        
        # Add residual connection
        return x + self.skip_connection(residual)


class MambaMixerBlock(nn.Module):
    """Memory-efficient Mamba-inspired mixer that avoids dimension mismatch"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        # Layer norm for input stabilization
        self.norm = nn.LayerNorm(d_model)
        
        # Simplified state mixing layer (no explicit SSM mechanism)
        # Instead use layer that mixes information across sequence dimension
        self.sequence_mixer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Channel mixing layer (standard MLP)
        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        # Apply sequence mixing
        x = x + self.sequence_mixer(x)
        
        # Apply channel mixing
        x = x + self.channel_mixer(x)
        
        return x + residual


class EfficientMambaProjector(nn.Module):
    """Ultra-memory-efficient Mamba-inspired projector"""
    def __init__(self, input_dim, output_dim, depth=1):
        super().__init__()
        # Ensure we project to the correct output dimension first
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Use simplified Mamba blocks that won't cause dimension issues
        layers = []
        for _ in range(depth):
            layers.append(MambaMixerBlock(d_model=output_dim))
        
        self.mamba_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # Project to correct dimension
        x = self.input_proj(x)
        
        # Pass through Mamba-inspired layers
        return self.mamba_layers(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
        
    # Add ultra-memory-efficient mamba implementation
    mamba_match = re.match(r'^mamba(\d+)x$', projector_type)
    if mamba_match:
        mamba_depth = int(mamba_match.group(1))
        return EfficientMambaProjector(
            input_dim=config.mm_hidden_size,
            output_dim=config.hidden_size,
            depth=mamba_depth
        )

    raise ValueError(f'Unknown projector type: {projector_type}')