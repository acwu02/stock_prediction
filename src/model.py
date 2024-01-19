from utils.dataset import StockDataset

import os
import sys
import pandas as pd
import numpy as np
import matplotlib

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, learning_rate, batch_size, epochs, train_data, test_data):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, 60)
        )

        self.model = self.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def train_model(self):
        # TODO
        size = len(self.train_dataloader)
        for batch, (X, y) in enumerate(self.train_dataloader):

            pred = self.model(X.to(torch.float32))
            loss = self.loss_fn(pred, y.to(torch.long))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_model(self):
        # TODO
        test_size = len(self.test_data)

    # For debugging
    def print_dataset(self, dataloader):
        for batch_num, (X, y) in enumerate(dataloader):
            print(f"Batch {batch_num}:", X, y)

    # For debugging
    def plot_dataset(self, dataloader):
        features = np.array([])
        labels = np.array([])
        for batch_num, (X, y) in enumerate(dataloader):
            features = np.append(features, X.numpy())
            labels = np.append(labels, y.numpy())
        print("Feature mean:", np.mean(features))
        print("Label mean:", np.mean(labels))
        print("Feature standard deviation:", np.std(features))
        print("Label standard deviation:", np.std(labels))

    # For debugging
    def print_layers(self):
        for layer in self.linear_relu_stack:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                print(layer.weight)
                print(layer.weight.dtype)

# Example usage
if __name__ == "__main__":
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    train_path = f'{os.getcwd()}/datasets/training/a.us.training.csv'
    test_path = f'{os.getcwd()}/datasets/testing/a.us.testing.csv'
    train_data = StockDataset(pd.read_csv(train_path))
    test_data = StockDataset(pd.read_csv(test_path))
    model = NeuralNetwork(learning_rate, batch_size, epochs, train_data, test_data)
    if len(sys.argv) > 1:
        if sys.argv[1] == "-pr":
            print("TRAINING:")
            model.print_dataset(model.train_dataloader)
            print("TESTING:")
            model.print_dataset(model.test_dataloader)
        elif sys.argv[1] == "-pl":
            print("TRAINING:")
            model.plot_dataset(model.train_dataloader)
            print("TESTING:")
            model.plot_dataset(model.test_dataloader)
        elif sys.argv[1] == "-l":
            model.print_layers()
    else:
         model.train_model()