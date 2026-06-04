import torch
import torch.nn as nn

class MorseTCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv2d = nn.Sequential(
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

        self.conv1d = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1, dilation=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, 3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, 3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv2d(x)
        x = x.squeeze(2)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        return x.permute(1, 0, 2)