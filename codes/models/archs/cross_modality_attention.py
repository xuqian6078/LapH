import torch
from torch import nn

class CrossModalityAttention(nn.Module):
    """Channel attention across modalities.

    Args:
        channels (int): Number of channels for each modality feature.
        reduction (int): Reduction ratio for hidden channels. Default: 16.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels * 2 // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, vi_feat: torch.Tensor, ir_feat: torch.Tensor):
        """Apply attention across two modality features."""
        concat = torch.cat([vi_feat, ir_feat], dim=1)
        att = self.avg_pool(concat)
        att = self.fc(att)
        att_vi, att_ir = torch.split(att, vi_feat.size(1), dim=1)
        return vi_feat * att_vi, ir_feat * att_ir
