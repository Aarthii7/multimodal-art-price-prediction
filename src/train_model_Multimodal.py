import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import AuctionDataset
from model import MultimodalModel

# =========================
# Load Dataset
# =========================
dataset = AuctionDataset(
    csv_path="data/auction_final.csv",
    image_folder="images"
)

print("Total samples:", len(dataset))

# =========================
# Train / Validation Split (80/20)
# =========================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =========================
# Get Category Sizes
# =========================
num_categories = []
for col in dataset.categorical_cols:
    num_categories.append(dataset.df[col].nunique())

# =========================
# Initialize Model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultimodalModel(num_categories).to(device)

# =========================
# Loss & Optimizer
# =========================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# =========================
# Training Loop
# =========================
epochs = 5

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

    # =========================
    # Validation
    # =========================
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, numeric, categorical, text, labels in val_loader:

            images = images.to(device)
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            text = text.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, numeric, categorical, text)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )