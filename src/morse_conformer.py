import torch
import torch.nn as nn
import copy
from torchaudio.models import Conformer

class MorseConformer2D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveMaxPool2d((1, None)),
        )
        
        input_dim = 64

        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=4,
            ffn_dim=256,
            num_layers=4,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
        )

        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x, lengths=None):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.transpose(1, 2)
        
        if lengths is not None:
            lengths = lengths // 4
        else:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
            
        x, _ = self.conformer(x, lengths)
        x = self.classifier(x)
        return x.transpose(0, 1)