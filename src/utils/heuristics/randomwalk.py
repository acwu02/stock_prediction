from heuristic import Heuristic
import random
import os

# Random walk heuristic, based on closing price of previous day
class RandomWalkHeuristic(Heuristic):
    def __init__(self, test_path, output_path, window):
        super().__init__(test_path, output_path)
        self.window = window

    def estimate_heuristic(self, index):
        data_point = self.test_df.iloc[index]['Close']
        prev_data_point = self.test_df.iloc[index - 1]['Close']
        lower_bound = prev_data_point - self.window
        upper_bound = prev_data_point + self.window
        estimated_point = random.uniform(lower_bound, upper_bound)
        diff = abs(data_point - estimated_point)
        return estimated_point

# Example usage
if __name__ == "__main__":
    window = 10
    train_path = f'{os.getcwd()}/datasets/training/a.us.training.csv'
    test_path = f'{os.getcwd()}/datasets/testing/a.us.testing.csv'
    output_path = f'{os.getcwd()}/results/a.us.results.random-heuristic.csv'
    heuristic = RandomWalkHeuristic(test_path, output_path, window)
    heuristic.evaluate()
