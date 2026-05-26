import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.feature_extraction.text import TfidfVectorizer


class AuctionDataset(Dataset):

    def __init__(self, csv_path, image_folder):
        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder

        # ======================
        # CLEAN PRICE
        # ======================
        self.df['price'] = (
            self.df['price']
            .astype(str)
            .str.replace('USD', '', regex=False)
            .str.replace('.', '', regex=False)   # remove thousand dot
            .str.strip()
        )

        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')

        # ======================
        # CLEAN YEAR
        # ======================
        self.df['yearCreation'] = pd.to_numeric(
            self.df['yearCreation'],
            errors='coerce'
        )

        # Normalize year
        self.df['yearCreation'] = (
             self.df['yearCreation'] - self.df['yearCreation'].mean()
        ) / self.df['yearCreation'].std()

        # Drop invalid rows
        self.df = self.df.dropna(subset=['price', 'yearCreation'])

        # ======================
        # LOG TRANSFORM PRICE
        # ======================
        self.df['log_price'] = torch.log1p(
            torch.tensor(self.df['price'].values, dtype=torch.float32)
        )

        # ======================
        # TF-IDF FOR TITLES
        # ======================
        self.df['title'] = self.df['title'].fillna("")

        self.vectorizer = TfidfVectorizer(max_features=300)
        self.text_features = self.vectorizer.fit_transform(
            self.df['title']
        ).toarray()

        # ======================
        # IMAGE TRANSFORM
        # ======================
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # ======================
        # FEATURES
        # ======================
        self.numeric_cols = ['yearCreation']
        self.categorical_cols = ['artist', 'signed', 'condition', 'period', 'movement']
        for col in self.categorical_cols:
            self.df[col] = self.df[col].astype('category').cat.codes

        self.df = self.df.reset_index(drop=True)


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = row['image_path']
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Numeric
        numeric = torch.tensor(
            row[self.numeric_cols].astype(float).values,
            dtype=torch.float32
        )

        # Categorical
        categorical = torch.tensor(
            row[self.categorical_cols].astype(int).values,
            dtype=torch.long
        )

        # Text (TF-IDF)
        text = torch.tensor(
            self.text_features[idx],
            dtype=torch.float32
        )

        # Label
        label = torch.tensor(row['log_price'], dtype=torch.float32)
        
        return image, numeric, categorical, text, label