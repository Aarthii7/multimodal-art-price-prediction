import torch
import torch.nn as nn
import torchvision.models as models


class SimpleFusionModel(nn.Module):
    def __init__(self, num_categories, text_dim=300):
        super(SimpleFusionModel, self).__init__()

        # Image Encoder
        self.image_encoder = models.efficientnet_b0(pretrained=True)
        self.image_encoder.classifier = nn.Identity()

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # Metadata Embeddings
        embedding_dim = 16
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, embedding_dim)
            for num_cat in num_categories
        ])

        # Metadata MLP
        metadata_input_dim = embedding_dim * len(num_categories) + 1

        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_input_dim, 64),
            nn.ReLU()
        )

        # Text MLP
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU()
        )

        # Final Fusion (Simple Concatenation)
        fusion_dim = 1280 + 64 + 128

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, image, numeric, categorical, text):

        # Image features
        image_feat = self.image_encoder(image)

        # Metadata features
        embedded = [
            emb(categorical[:, i])
            for i, emb in enumerate(self.embeddings)
        ]

        embedded = torch.cat(embedded, dim=1)
        metadata_input = torch.cat([numeric, embedded], dim=1)
        metadata_feat = self.metadata_mlp(metadata_input)

        # Text features
        text_feat = self.text_mlp(text)

        # Simple Concatenation
        fused = torch.cat([image_feat, metadata_feat, text_feat], dim=1)

        output = self.regressor(fused)

        return output