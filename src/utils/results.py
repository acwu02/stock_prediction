import csv
from torch import nn

class Results():
    def __init__(self, path):
        self.path = path
        self.header = ['Actual', 'Predicted', 'AbsDiff']
        self.write_header()

    def write_header(self):
        with open(self.path, 'w', newline='') as output:
            csv_writer = csv.writer(output)
            csv_writer.writerow(self.header)

    def write(self, actual, predicted):
        with open(self.path, 'a', newline='') as output:
            csv_writer = csv.writer(output)
            diff = abs(actual - predicted)
            csv_writer.writerow([actual, predicted, diff])

