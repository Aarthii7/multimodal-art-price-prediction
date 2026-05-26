import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import AuctionDataset
from model import MultimodalModel
from model_simple import SimpleFusionModel
from metadata_model import MetadataOnlyModel
from image_model import ImageOnlyModel
from text_model import TextOnlyModel


# =========================
# Load Dataset
# =========================
dataset = AuctionDataset(
    csv_path="data/auction_final.csv",
    image_folder="images"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 80-20 split
train_idx, val_idx = train_test_split(
    range(len(dataset)), test_size=0.2, random_state=42
)

train_loader = DataLoader(
    torch.utils.data.Subset(dataset, train_idx),
    batch_size=16, shuffle=True
)

val_loader = DataLoader(
    torch.utils.data.Subset(dataset, val_idx),
    batch_size=16, shuffle=False
)

num_categories = [
    dataset.df[col].nunique()
    for col in dataset.categorical_cols
]


def train_and_predict(model, model_name):

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5
    )

    epochs = 10

    # Train
    for epoch in range(epochs):
        model.train()
        for images, numeric, categorical, text, labels in train_loader:

            images = images.to(device)
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            text = text.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            if model_name == "image":
                outputs = model(images)
            elif model_name == "metadata":
                outputs = model(numeric, categorical)
            elif model_name == "text":
                outputs = model(text)
            else:
                outputs = model(images, numeric, categorical, text)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Predict
    model.eval()
    preds = []
    true_vals = []

    with torch.no_grad():
        for images, numeric, categorical, text, labels in val_loader:

            images = images.to(device)
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            text = text.to(device)

            if model_name == "image":
                outputs = model(images)
            elif model_name == "metadata":
                outputs = model(numeric, categorical)
            elif model_name == "text":
                outputs = model(text)
            else:
                outputs = model(images, numeric, categorical, text)

            preds.extend(outputs.cpu().numpy().flatten())
            true_vals.extend(labels.numpy().flatten())

    preds_exp = np.expm1(np.array(preds))
    true_exp = np.expm1(np.array(true_vals))

    return true_exp, preds_exp


# =========================
# Train All Models
# =========================

true_price, image_preds = train_and_predict(
    ImageOnlyModel(), "image"
)

_, metadata_preds = train_and_predict(
    MetadataOnlyModel(num_categories), "metadata"
)

_, text_preds = train_and_predict(
    TextOnlyModel(), "text"
)

_, simple_preds = train_and_predict(
    SimpleFusionModel(num_categories), "simple"
)

_, attention_preds = train_and_predict(
    MultimodalModel(num_categories), "attention"
)


# =========================
# Save CSV
# =========================

results_df = pd.DataFrame({
    "True Price": true_price,
    "Image Prediction": image_preds,
    "Metadata Prediction": metadata_preds,
    "Text Prediction": text_preds,
    "Simple Fusion Prediction": simple_preds,
    "Attention Fusion Prediction": attention_preds
})

os.makedirs("results", exist_ok=True)
results_df.to_csv("results/predictions.csv", index=False)

print("Predictions saved to results/predictions.csv")