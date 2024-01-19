import pandas as pd
import numpy as np

import os
import torch

from torch.utils.data import Dataset, DataLoader

# Wrapper for torch Dataset class
class StockDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.features = self.df.iloc[:, :-1]
        self.labels = self.df.iloc[:, -1]
        # We calculate the mean of the dataset to normalize the data points over.
        # This is because stock prices generally increase over time,
        # so our testing points may not be adequately represented in the historic data.
        self.mean = df.mean().mean()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = self.normalize(self.features.iloc[idx].values)
        labels = self.normalize(self.labels.iloc[idx])
        return features, labels

    def normalize(self, tensor):
        return (tensor - self.mean) / self.mean

# Example usage
if __name__ == "__main__":
    TRAINING_DIR = f'{os.getcwd()}/datasets/training'
    csv_file = 'a.us.training.csv'
    df = pd.read_csv(f'{TRAINING_DIR}/{csv_file}')
    dataset = StockDataset(df)
    batch_size = 54
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)