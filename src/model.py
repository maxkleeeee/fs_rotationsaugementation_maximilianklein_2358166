"""LSTM-Klassifikator fuer isolierte Gebaerden."""
from __future__ import annotations

import torch.nn as nn


class SignLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes,
                 dropout, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        d = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    def forward(self, x):
        out, _ = self.lstm(x)
        # Mittelwert ueber alle Zeitschritte statt nur des letzten
        return self.head(out.mean(dim=1))
