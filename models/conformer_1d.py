import torch
import torch.nn as nn
from torchaudio.models import Conformer


class MorseConformer1D(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freq_bins: int = 70,
        kernel_size: int = 7,
        channels: list[int] = [64, 128],
        num_heads: int = 4,
        ffn_dim: int = 512,
        num_layers: int = 4,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        in_channels = freq_bins
        padding = kernel_size // 2

        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        input_dim = channels[-1]

        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = x.squeeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        x, _ = self.conformer(x, lengths)
        x = self.norm(x)
        
        x = self.classifier(x)
        return x.transpose(0, 1)