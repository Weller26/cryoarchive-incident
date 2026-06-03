import torch
import torch.nn as nn


class MorseCRNN1D(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_type: str = 'gru',
        freq_bins: int = 70,
        kernel_size: int = 7,
        channels: list[int] = [64, 128],
        hidden_size: int = 160,
        num_layers: int = 2,
        bidirectional: bool = True,
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
        self.out_channels = in_channels
        self.cnn = nn.Sequential(*layers)

        if model_type.lower() == 'gru':
            self.crnn = nn.GRU(
                input_size=self.out_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        elif model_type.lower() == 'lstm':
            self.crnn = nn.LSTM(
                input_size=self.out_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        else:
            raise NameError(f'Имя модели {model_type} невалидно')

        crnn_out = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(crnn_out)
        self.classifier = nn.Linear(crnn_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        x, _ = self.crnn(x)
        x = self.norm(x)
        
        x = self.classifier(x)
        return x.transpose(0, 1)