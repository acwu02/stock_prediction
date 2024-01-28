import csv
import pandas as pd
import numpy as np
from src.utils.results import Results
from torch import nn
from torch import Tensor

# Populates a .csv result table with a given heuristic as a benchmark to compare the model with
class Heuristic():
    def __init__(self, test_path, output_path):
        self.test_path = test_path
        self.test_df = pd.read_csv(self.test_path)
        self.output_path = output_path
        self.actual = []
        self.predicted = []
        self.results = Results(self.output_path)
        self.loss_fn = nn.MSELoss()

    # estimate output of data point at given index
    def estimate_heuristic(self, index):
        pass

    def evaluate(self):
        size = len(self.test_df)
        self.features = self.test_df.iloc[:, :-1]
        self.labels = self.test_df.iloc[:, -1]
        for index, row in self.features.iterrows():
            pred = self.estimate_heuristic(index)
            y = self.labels.iloc[index]
            self.results.write(y, pred)
        assert len(self.actual) == len(self.predicted)
        self.calculate_loss()

    def calculate_loss(self):
        self.results_df = pd.read_csv(self.output_path)
        pred_vals = self.results_df.iloc[:, 0]
        y_vals = self.results_df.iloc[:, 1]
        loss = self.loss_fn(Tensor(pred_vals), Tensor(y_vals))
        print(f'Mean squared error loss: {loss}')
