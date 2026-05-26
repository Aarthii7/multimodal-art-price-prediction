import torch
import torch.nn as nn
import torchvision.models as models


class ImageOnlyModel(nn.Module):
    def __init__(self):
        super(ImageOnlyModel, self).__init__()

        self.image_encoder = models.efficientnet_b0(pretrained=True)
        self.image_encoder.classifier = nn.Identity()

        # Freeze backbone (data-efficient setup)
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, image):

        features = self.image_encoder(image)
        output = self.regressor(features)

        return output