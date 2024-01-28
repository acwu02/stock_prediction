from heuristic import Heuristic
import pandas as pd
import numpy as np
import csv
import os

# Linear regression heuristic, assuming stock price relationship is linear
class LinearRegressionHeuristic(Heuristic):
    def __init__(self, train_path, test_path, output_path, use_np):
        super().__init__(test_path, output_path)
        self.train_path = train_path
        self.train_df = pd.read_csv(self.train_path)
        self.test_features = self.test_df.iloc[:, :-1]
        self.test_labels = self.test_df.iloc[:, -1]
        self.train_features = self.train_df.iloc[:, :-1]
        self.train_labels = self.train_df.iloc[:, -1]
        # matrix of parameters
        self.beta = np.zeros(self.train_df.iloc[:, :-1].shape)
        self.use_np = use_np
        self.linear_regression = self.linear_regression_np if self.use_np else self.linear_regression_native

    def estimate_heuristic(self, index):
        data_point = self.train_features.iloc[index].to_numpy()
        return np.dot(data_point, self.beta)

    def linear_regression_native(self):
        X = self.test_features.values
        Y = self.test_labels.values
        # using least squares method to estimate parameters
        self.beta = np.dot(np.linalg.inv(np.dot(X.T, X)),
                      np.dot(X.T, Y))
        print("Weights trained")
        self.evaluate()
        print(f"Results stored in {self.output_path}")

    def linear_regression_np(self):
        X = self.test_features.values
        Y = self.test_labels.values
        coefficients, residuals, rank, singular_values = np.linalg.lstsq(X, Y, rcond=None)
        self.beta = coefficients
        print("Weights trained")
        self.evaluate()
        print(f"Results stored in {self.output_path}")

# Example usage
if __name__ == "__main__":
    train_path = f'{os.getcwd()}/datasets/training/a.us.training.csv'
    test_path = f'{os.getcwd()}/datasets/testing/a.us.testing.csv'
    output_path = f'{os.getcwd()}/results/a.us.results.linreg-heuristic.csv'
    heuristic = LinearRegressionHeuristic(train_path, test_path, output_path, use_np=True)
    heuristic.linear_regression()
    heuristic.evaluate()


