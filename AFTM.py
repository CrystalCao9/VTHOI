"""
AFTM is the visual feature enhancement module applied following feature extraction by the BLIP visual encoder.
"""

import torch
import torch.nn as nn


class AFTM(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature projection
        self.feature_proj = nn.Linear(1024, 256)

        # Swin parameters
        self.swin_dim = 256
        self.swin_depth = 1
        self.num_heads = 256 // 32
        self.window_size = 8
        self.base_sd = 0.2
        self.mlp_expansion = 4

        # Build Swin blocks
        self.shift_pattern = [
            [self.window_size // 2] * 2 if i % 2 else [0, 0]
            for i in range(self.swin_depth)
        ]
        self.stochastic_probs = (torch.linspace(0, 1, 12)[10 - self.swin_depth:10] * self.base_sd).tolist()

        # Swin components
        self.attn_blocks = nn.Sequential(*[
            self.create_swin_block(shift=self.shift_pattern[i], sd_prob=self.stochastic_probs[i])
            for i in range(self.swin_depth)
        ])

        # MLP components
        self.mlp_expansion = 4
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.swin_dim) for _ in range(2 * self.swin_depth)
        ])

    def create_swin_block(self, shift: list, sd_prob: float):
        return nn.Sequential(
            # Attention module
            nn.Sequential(
                nn.LayerNorm(self.swin_dim),
                self.WindowAttention(
                    dim=self.swin_dim,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    shift=shift
                ),
                nn.Dropout(sd_prob)
            ),
            # MLP module
            nn.Sequential(
                nn.LayerNorm(self.swin_dim),
                nn.Linear(self.swin_dim, self.swin_dim * self.mlp_expansion),
                nn.GELU(),
                nn.Linear(self.swin_dim * self.mlp_expansion, self.swin_dim),
                nn.Dropout(sd_prob)
            )
        )

    class WindowAttention(nn.Module):
        """Simplified window attention module"""

        def __init__(self, dim: int, num_heads: int, window_size: int, shift: list):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.window = window_size
            self.shift = shift

            # Attention parameters
            self.qkv_proj = nn.Linear(dim, dim * 3)
            self.scale = (dim // num_heads) ** -0.5

        def forward(self, x: torch.Tensor):
            # Simplified attention implementation
            B, H, W, C = x.shape
            qkv = self.qkv_proj(x).reshape(B, H, W, 3, self.num_heads, C // self.num_heads)
            q, k, v = qkv.permute(3, 0, 4, 1, 2, 5).unbind(0)  # [3, B, h, H, W, c]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

            return (attn @ v).transpose(1, 2).reshape(B, H, W, C)

    def forward(self, x: torch.Tensor):
        # Initial processing
        batch_size = x.size(0)
        spatial_feat = x[:, 1:].reshape(batch_size, 14, 14, -1)
        proj_feat = self.feature_proj(spatial_feat)

        # Swin processing
        x = proj_feat
        for i, block in enumerate(self.attn_blocks):
            # Attention path
            attn_out = block[0](x)
            x = x + attn_out

            # MLP path
            mlp_out = block[1](x)
            x = x + mlp_out

        # Final output
        return x.reshape(batch_size, 196, -1)