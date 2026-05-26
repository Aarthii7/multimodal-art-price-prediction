import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import AuctionDataset
from model import MultimodalModel


# =========================
# Load Dataset
# =========================
dataset = AuctionDataset(
    csv_path="data/auction_final.csv",
    image_folder="images"
)

print("Total clean samples:", len(dataset))

print("Sample cleaned prices:")
print(dataset.df['price'].head(20))
print("Max price:", dataset.df['price'].max())
print("Min price:", dataset.df['price'].min())
print("Average price:", dataset.df['price'].mean())

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Category Sizes
# =========================
num_categories = [
    dataset.df[col].nunique()
    for col in dataset.categorical_cols
]

# =========================
# K-Fold Setup
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
print("Average price:", dataset.df['price'].mean())
print("Price std:", dataset.df['price'].std())

# =========================
# Cross Validation Loop
# =========================
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

    print(f"\n========== Fold {fold+1} ==========")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

    model = MultimodalModel(num_categories).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5
    )

    epochs = 20

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, numeric, categorical, text, labels in train_loader:

            images = images.to(device)
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            text = text.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images, numeric, categorical, text)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}")

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    preds = []
    true_vals = []

    with torch.no_grad():
        for images, numeric, categorical, text, labels in val_loader:

            images = images.to(device)
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            text = text.to(device)

            outputs = model(images, numeric, categorical, text)

            preds.extend(outputs.cpu().numpy().flatten())
            true_vals.extend(labels.numpy().flatten())

    preds = np.array(preds)
    true_vals = np.array(true_vals)

    # Convert back from log scale
    preds_exp = np.expm1(preds)
    true_exp = np.expm1(true_vals)

    mae = mean_absolute_error(true_exp, preds_exp)
    rmse = np.sqrt(mean_squared_error(true_exp, preds_exp))
    r2 = r2_score(true_exp, preds_exp)

    print(f"Fold {fold+1} Results → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

    fold_results.append((mae, rmse, r2))


# =========================
# Average Results
# =========================
fold_results = np.array(fold_results)

avg_mae = fold_results[:, 0].mean()
avg_rmse = fold_results[:, 1].mean()
avg_r2 = fold_results[:, 2].mean()

print("\n========== Final Average Performance ==========")

print(f"Average MAE: {avg_mae:.2f}")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average R2: {avg_r2:.4f}")