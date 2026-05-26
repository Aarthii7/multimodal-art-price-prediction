import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import AuctionDataset
from metadata_model import MetadataOnlyModel


dataset = AuctionDataset(
    csv_path="data/auction_final.csv",
    image_folder="images"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_categories = [
    dataset.df[col].nunique()
    for col in dataset.categorical_cols
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

    print(f"\n========== Fold {fold+1} ==========")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

    model = MetadataOnlyModel(num_categories).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    epochs = 20

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for images, numeric, categorical, text, labels in train_loader:

            numeric = numeric.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(numeric, categorical)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

    # Validation
    model.eval()
    preds = []
    true_vals = []

    with torch.no_grad():
        for images, numeric, categorical, text, labels in val_loader:

            numeric = numeric.to(device)
            categorical = categorical.to(device)

            outputs = model(numeric, categorical)

            preds.extend(outputs.cpu().numpy().flatten())
            true_vals.extend(labels.numpy().flatten())

    preds = np.array(preds)
    true_vals = np.array(true_vals)

    preds_exp = np.expm1(preds)
    true_exp = np.expm1(true_vals)

    mae = mean_absolute_error(true_exp, preds_exp)
    rmse = np.sqrt(mean_squared_error(true_exp, preds_exp))
    r2 = r2_score(true_exp, preds_exp)

    print(f"Fold {fold+1} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

    fold_results.append((mae, rmse, r2))


fold_results = np.array(fold_results)

print("\n===== Metadata Only Average =====")
print("MAE:", fold_results[:, 0].mean())
print("RMSE:", fold_results[:, 1].mean())
print("R2:", fold_results[:, 2].mean())