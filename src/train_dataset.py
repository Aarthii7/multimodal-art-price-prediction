from dataset import AuctionDataset

dataset = AuctionDataset(
    csv_path="data/auction_final.csv",
    image_folder="images"
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("Image shape:", sample[0].shape)
print("Numeric:", sample[1])
print("Categorical:", sample[2])
print("Text shape:", sample[3].shape)
print("Label:", sample[4])