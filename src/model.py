from src.utils.dataset import StockDataset
from src.utils.results import Results

import os
import sys
import pandas as pd
import numpy as np
import matplotlib

import torch
from torch import nn
from torch.utils.data import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(
            self,
            learning_rate,
            batch_size,
            epochs,
            train_data,
            test_data,
            offset,
            weight_path,
            model_results_path,
            heuristic_results_path
        ):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        self.train_mean = train_data.mean
        self.test_mean = test_data.mean
        self.offset = offset

        self.weight_path = weight_path
        self.model_results_path = model_results_path
        self.heuristic_results_path = heuristic_results_path

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=60, out_features=30),
            nn.ReLU(),
            nn.Linear(in_features=30, out_features=15),
            nn.ReLU(),
            nn.Linear(in_features=15, out_features=1)
        )

        model = self.to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def train_loop(self):
        size = len(self.train_dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            pred = model(X.to(torch.float32))
            loss = self.loss_fn(pred, y.to(torch.float32))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self):
        size = len(self.test_dataloader.dataset)
        model.eval()
        num_batches = len(self.test_dataloader)
        avg_loss = 0
        # TODO fix data types of dataset
        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = model(X.to(torch.float32))
                avg_loss += self.loss_fn(pred, y.to(torch.float32)).item()
        avg_loss /= num_batches
        print(f"Test Error: \n Avg loss: {avg_loss:>8f} \n")

    def begin_training(self):
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop()
            self.test_loop()
        print("Done!")
        self.save_weights()

    def save_weights(self):
        # TODO save weights
        torch.save(self.state_dict(), self.weight_path)
        print(f'Weights saved to {self.weight_path}')

    def load_weights(self):
        model.load_state_dict(torch.load(self.weight_path))
        model.eval()

    def evaluate(self):
        print("Evaluating model:")
        self.load_weights()
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        results = Results(self.model_results_path)
        avg_loss = 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = model(X.to(torch.float32))
                # squeezing to 1D tensor
                pred = pred.squeeze(dim=1)
                for i, row in enumerate(pred):
                    results.write(self.reverse_normalization(y[i].item(), self.test_mean, self.offset),
                                  self.reverse_normalization(row.item(), self.test_mean, self.offset))
                avg_loss += self.loss_fn(pred, y)
        avg_loss /= size
        print(f'Model evaluated; results saved to {self.model_results_path}')
        # TODO replace below with printing MSE loss of all predictions and all labels
        print(f'Mean squared error loss: {avg_loss}')

    def reverse_normalization(self, tensor, mean, offset):
        return (tensor - offset) * mean + mean

    # For debugging
    def print_dataset(self, dataloader):
        for batch_num, (X, y) in enumerate(dataloader):
            print(f"Batch {batch_num}:\n", "Features:\n", X, "\nLabels:\n", y)

    # For debugging
    def print_reversed_dataset(self, dataloader, mean, offset):
        for batch_num, (X, y) in enumerate(dataloader):
            print(f"Batch {batch_num}:\n",
                "Features:\n", self.reverse_normalization(X, mean, offset),
                "\nLabels:\n", self.reverse_normalization(y, mean, offset))

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
    epochs = 100
    offset = 5
    # TODO clean up os.getcwd, init env variables
    train_path = f'{os.getcwd()}/datasets/training/a.us.training.csv'
    test_path = f'{os.getcwd()}/datasets/testing/a.us.testing.csv'
    weight_path = f'{os.getcwd()}/weights/weights.pt'
    model_results_path = f'{os.getcwd()}/results/a.us.results-model.csv'
    heuristic_results_path = f'{os.getcwd()}/results/a.us.results-heuristic.csv'
    train_data = StockDataset(pd.read_csv(train_path), offset)
    test_data = StockDataset(pd.read_csv(test_path), offset)
    model = NeuralNetwork(
        learning_rate,
        batch_size,
        epochs,
        train_data,
        test_data,
        offset,
        weight_path,
        model_results_path,
        heuristic_results_path).to(device)
    # TODO clean up flags
    if len(sys.argv) > 1:
        if sys.argv[1] == "--print":
            print("TRAINING:\n")
            model.print_dataset(model.train_dataloader)
            print("TESTING:\n")
            model.print_dataset(model.test_dataloader)
        elif sys.argv[1] == "--print-reversed":
            print("TRAINING:\n")
            model.print_reversed_dataset(model.train_dataloader, model.train_mean)
            print("TESTING:\n")
            model.print_reversed_dataset(model.test_dataloader, model.test_mean)
        elif sys.argv[1] == "--plot":
            print("TRAINING:")
            model.plot_dataset(model.train_dataloader)
            print("TESTING:")
            model.plot_dataset(model.test_dataloader)
        elif sys.argv[1] == "--print-layers":
            model.print_layers()
        elif sys.argv[1] == "--evaluate":
            model.evaluate()
    else:
         model.begin_training()