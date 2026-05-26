import torch
import torch.nn as nn


class MetadataOnlyModel(nn.Module):
    def __init__(self, num_categories):
        super(MetadataOnlyModel, self).__init__()

        embedding_dim = 16

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, embedding_dim)
            for num_cat in num_categories
        ])

        metadata_input_dim = embedding_dim * len(num_categories) + 1

        self.mlp = nn.Sequential(
            nn.Linear(metadata_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, numeric, categorical):

        embedded = [
            emb(categorical[:, i])
            for i, emb in enumerate(self.embeddings)
        ]

        embedded = torch.cat(embedded, dim=1)

        metadata_input = torch.cat([numeric, embedded], dim=1)

        output = self.mlp(metadata_input)

        return output