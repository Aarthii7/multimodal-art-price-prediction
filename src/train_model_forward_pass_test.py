import torch
from dataset import AuctionDataset
from model import MultimodalModel

dataset = AuctionDataset(
    csv_path="data/auction_final.csv",
    image_folder="images"
)

num_categories = []

for col in dataset.categorical_cols:
    num_categories.append(
        dataset.df[col].nunique()
    )

print("Category sizes:", num_categories)

model = MultimodalModel(num_categories)

sample = dataset[0]

image = sample[0].unsqueeze(0)
numeric = sample[1].unsqueeze(0)
categorical = sample[2].unsqueeze(0)
text = sample[3].unsqueeze(0)

output = model(image, numeric, categorical, text)

print("Model output shape:", output.shape)