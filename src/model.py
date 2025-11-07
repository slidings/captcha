# src/model.py
import torch
import torch.nn as nn
from typing import Tuple
from .vocab import BLANK, ITOCH


class ResidualBlock(nn.Module):
    """Standard ResNet-style residual block (no downsampling)."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

        # Initialize the second BN's gamma to 0 (identity start)
        nn.init.constant_(self.bn2.weight, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)  # standard ResNet post-add ReLU
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            #nn.InstanceNorm2d(cout, affine=True),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class CRNN(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 input_channels: int,
                 img_height: int,
                 cnn_out: int = 256, 
                 lstm_hidden: int = 256, 
                 lstm_layers: int = 2, 
                 dropout: float = 0.3):
        
        super().__init__()
        
        self.cnn = nn.Sequential(
            ConvBlock(input_channels, 64), nn.MaxPool2d(2, 2),  # H/2
            ConvBlock(64, 128),            nn.MaxPool2d(2, 2),  # H/4
            ResidualBlock(128),
            ConvBlock(128, cnn_out),
        )
        
        downsampled_height = img_height // 4 
        lstm_input_size = cnn_out * downsampled_height 

        self.bi_lstm = nn.LSTM(input_size=lstm_input_size,
                               hidden_size=lstm_hidden,
                               num_layers=lstm_layers, bidirectional=True, 
                               batch_first=True, dropout=dropout)
                               
        #self.classifier = nn.Linear(2*lstm_hidden, num_classes)

        #regularisation if necessary
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2*lstm_hidden, num_classes)
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.cnn(images)
        B, Cf, Hp, Wp = feat.shape

        feat = feat.permute(0, 3, 1, 2).contiguous()
        feat = feat.view(B, Wp, -1)
        
        seq = feat 

        out, _ = self.bi_lstm(seq)
        logits = self.classifier(out)

        logits = logits.permute(1, 0, 2).contiguous()

        input_lengths = torch.full((B,), fill_value=Wp, dtype=torch.long, device=images.device)
        return logits, input_lengths
