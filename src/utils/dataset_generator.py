import pandas as pd
import os
import csv
import math

WINDOW_SIZE = 15
WORKING_DIR = os.getcwd()

# Handles generation of .csv datasets for an individual stock by building feature vector of lagged values
# with labeled values
# Run this file to generate datasets
class DatasetGenerator():
    def __init__(self, input_path, training_path, testing_path, label):
        self.input_path = input_path
        self.training_path = training_path
        self.testing_path = testing_path
        self.label = label
        self.df = pd.read_csv(self.input_path)
        self.training_range_start = WINDOW_SIZE
        self.training_range_end = math.ceil(len(self.df) * 0.8)
        self.testing_range_start = self.training_range_end
        self.testing_range_end = len(self.df)
        self.features = self.df.columns.tolist()
        self.lagged_features = ['Open', 'High', 'Low', 'Close']
        self.header = self.get_header()

    def get_header(self):
        header = [f'{field}{i}' for i in reversed(range(1, WINDOW_SIZE + 1)) for field in self.lagged_features]
        header.append(self.label)
        return header

    def generate_datasets(self):
        self.write_to_output(self.training_path, self.training_range_start, self.training_range_end)
        self.write_to_output(self.testing_path, self.testing_range_start, self.testing_range_end)

    def write_to_output(self, output_path, start, end):
        with open(output_path, 'w', newline='') as output:
            csv_writer = csv.writer(output)
            csv_writer.writerow(self.header)
            for row_index in range(start, end):
                feature_vector = self.build_feature_vector(row_index)
                label_value = self.df.iloc[row_index][self.label]
                feature_vector.append(label_value)
                csv_writer.writerow(feature_vector)

    def build_feature_vector(self, row_index):
        start_index = max(0, row_index - WINDOW_SIZE)
        end_index = row_index
        feature_vector = []
        for lagged_feature in self.lagged_features:
            for i in range(start_index, end_index):
                if i < 0:
                    feature_vector.append(float('nan'))
                else:
                    feature_vector.append(self.df.iloc[i][lagged_feature])
        return feature_vector

# Example usage
if __name__ == '__main__':
    input_path = f'datasets/stock_data/Stocks/a.us.txt'
    training_path = f'datasets/training/a.us.training.csv'
    testing_path = f'datasets/testing/a.us.testing.csv'
    dataset = DatasetGenerator(input_path, training_path, testing_path, label='Close')
    dataset.generate_datasets()