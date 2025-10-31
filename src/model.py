# src/model.py
import torch
import torch.nn as nn
from typing import Tuple
from .vocab import BLANK, ITOCH

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class CRNN(nn.Module):
    def __init__(self, num_classes: int, cnn_out: int = 256, lstm_hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # CNN encoder: (B,3,H,W) -> (B,Cf,H',W')
        self.cnn = nn.Sequential(
            ConvBlock(3, 64),   nn.MaxPool2d(2, 2),      # /2
            ConvBlock(64,128),  nn.MaxPool2d(2, 2),      # /4
            ConvBlock(128,256),
            ConvBlock(256,cnn_out)
        )
        # BiLSTM over width dimension
        self.bi_lstm = nn.LSTM(input_size=cnn_out, hidden_size=lstm_hidden,
                               num_layers=lstm_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(2*lstm_hidden, num_classes)  # bidirectional ⇒ *2

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images: (B,C,H,W)
        returns:
            logits: (T, B, C) for CTCLoss
            input_lengths: (B,) time lengths per sample (after CNN downsample)
        """
        feat = self.cnn(images)           # (B,Cf,H',W')
        B, Cf, Hp, Wp = feat.shape

        # collapse height: average over H' (robust & simple)
        feat = feat.mean(dim=2)           # (B,Cf,W')

        # transpose to (B, W', Cf), feed to LSTM
        seq = feat.permute(0, 2, 1)       # (B, T, Cf) where T = W'

        out, _ = self.bi_lstm(seq)        # (B, T, 2*hidden)
        logits = self.classifier(out)     # (B, T, num_classes)

        # for CTCLoss: (T,B,C)
        logits = logits.permute(1, 0, 2).contiguous()

        # input lengths: whole width after downsample ~ Wp
        input_lengths = torch.full((B,), fill_value=Wp, dtype=torch.long, device=images.device)
        return logits, input_lengths
