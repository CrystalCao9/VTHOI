'''
ALFM is designed to fuse logits from two visual HOI detection branches
Input：[B,C,T]
Output：[2B,C,T]
'''

import torch
from torch import nn
import torch.nn.functional as F

class ALFM(nn.Module):
    def __init__(self, channels):
        super(ALFM, self).__init__()

        # Spatial attention convolution layers
        self.spatial_attn_conv1 = nn.Conv1d(2, 1, 3, padding=1)
        self.spatial_attn_conv2 = nn.Conv1d(1, 1, 3, padding=1)

        # Channel compression projection layers
        self.channel_avg_compress = nn.Conv1d(channels, channels // 2, 1)
        self.channel_max_compress = nn.Conv1d(channels, channels // 2, 1)

        # Channel expansion projection layers
        self.channel_avg_expand = nn.Conv1d(channels // 2, channels, 1)
        self.channel_max_expand = nn.Conv1d(channels // 2, channels, 1)

        # Parameter initialization
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.normal_(layer.weight, 0, 0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def _compute_channel_attn(self, feature, compress_layer, expand_layer):
        compressed = F.relu(compress_layer(feature.mean(dim=-1, keepdim=True)))
        return expand_layer(compressed)

    def forward(self, input_feat1, input_feat2):

        # Channel attention computation
        channel_attn1 = (self._compute_channel_attn(input_feat1, self.channel_avg_compress, self.channel_avg_expand) +
                         self._compute_channel_attn(input_feat1, self.channel_max_compress, self.channel_max_expand))

        channel_attn2 = (self._compute_channel_attn(input_feat2, self.channel_avg_compress, self.channel_avg_expand) +
                         self._compute_channel_attn(input_feat2, self.channel_max_compress, self.channel_max_expand))

        # Cross-attention mechanism
        scale = channel_attn1.size(1) ** 0.5
        cross_attn_matrix = torch.matmul(channel_attn1, channel_attn2.transpose(1, 2)) / scale
        attended_feat1 = torch.matmul(F.softmax(cross_attn_matrix, dim=-1), input_feat1)
        attended_feat2 = torch.matmul(F.softmax(cross_attn_matrix.transpose(1, 2), dim=-1), input_feat2)

        # Spatial mask generation
        def _generate_spatial_mask(feature):
            mean_pool = torch.mean(feature, dim=1, keepdim=True)
            max_pool, _ = torch.max(feature, dim=1, keepdim=True)
            concatenated = torch.cat([mean_pool, max_pool], dim=1)
            attn_weights = F.relu(self.spatial_attn_conv1(concatenated))
            return F.softmax(self.spatial_attn_conv2(attn_weights), dim=-1)

        spatial_mask1 = _generate_spatial_mask(attended_feat1)
        spatial_mask2 = _generate_spatial_mask(attended_feat2)

        # Feature enhancement and fusion
        enhanced_feat1 = input_feat1 * spatial_mask1 + input_feat1
        enhanced_feat2 = input_feat2 * spatial_mask2 + input_feat2

        return torch.cat([enhanced_feat1, enhanced_feat2], dim=0)
