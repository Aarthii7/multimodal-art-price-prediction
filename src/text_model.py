import torch
import torch.nn as nn


class TextOnlyModel(nn.Module):
    def __init__(self, text_dim=300):
        super(TextOnlyModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, text):
        return self.mlp(text)