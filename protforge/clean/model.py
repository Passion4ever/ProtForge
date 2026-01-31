"""CLEAN model architecture."""

import torch
import torch.nn as nn


class LayerNormNet(nn.Module):
    """CLEAN projection network: 1280 -> hidden -> out."""

    def __init__(self, hidden_dim, out_dim, device, dtype, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(self.dropout(self.ln1(self.fc1(x))))
        x = torch.relu(self.dropout(self.ln2(self.fc2(x))))
        return self.fc3(x)
