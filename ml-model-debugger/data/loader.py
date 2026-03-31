import pandas as pd
from sklearn.model_selection import train_test_split
import os

class DataLoader:
    def __init__(self, file_path: str, target_column: str, test_size: float = 0.2, random_state: int = 42):
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def load_and_split(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at: {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
