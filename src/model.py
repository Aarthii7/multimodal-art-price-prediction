import torch
import torch.nn as nn
import torchvision.models as models


class MultimodalModel(nn.Module):
    def __init__(self, num_categories, text_dim=300):
        super(MultimodalModel, self).__init__()

        # =========================
        # IMAGE ENCODER (EfficientNet-B0)
        # =========================
        from torchvision.models import EfficientNet_B0_Weights

        self.image_encoder = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        self.image_encoder.classifier = nn.Identity()

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        image_feature_dim = 1280

        # =========================
        # METADATA ENCODER
        # =========================
        embedding_dim = 16

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, embedding_dim)
            for num_cat in num_categories
        ])

        metadata_input_dim = embedding_dim * len(num_categories) + 1

        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # =========================
        # TEXT ENCODER (TF-IDF MLP)
        # =========================
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # =========================
        # PROJECTION TO SAME DIM
        # =========================
        self.common_dim = 256

        self.image_proj = nn.Linear(1280, self.common_dim)
        self.meta_proj = nn.Linear(64, self.common_dim)
        self.text_proj = nn.Linear(128, self.common_dim)

        # =========================
        # FUSION + ATTENTION
        # =========================
        fusion_dim = self.common_dim * 3

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, image, numeric, categorical, text):

        # Image
        image_feat = self.image_encoder(image)

        # Metadata
        embedded = [
            emb(categorical[:, i])
            for i, emb in enumerate(self.embeddings)
        ]

        embedded = torch.cat(embedded, dim=1)
        metadata_input = torch.cat([numeric, embedded], dim=1)
        metadata_feat = self.metadata_mlp(metadata_input)

        # Text
        text_feat = self.text_mlp(text)

        # Fusion
        # =========================
        # PROJECT TO COMMON DIM
        # =========================
        image_feat = self.image_proj(image_feat)
        metadata_feat = self.meta_proj(metadata_feat)
        text_feat = self.text_proj(text_feat)

        # =========================
        # STACK MODALITIES
        # =========================
        modalities = torch.stack(
        [image_feat, metadata_feat, text_feat],
        dim=1
        )

        # =========================
        # MODALITY ATTENTION
        # =========================
        attention_weights = torch.softmax(
        torch.mean(modalities, dim=2),
        dim=1
        ).unsqueeze(-1)

        weighted_modalities = modalities * attention_weights

        # Flatten
        fused = weighted_modalities.view(
        weighted_modalities.size(0), -1
        )

        # =========================
        # REGRESSION
        # =========================
        output = self.regressor(fused)
        return output