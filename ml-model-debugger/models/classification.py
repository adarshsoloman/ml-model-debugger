from sklearn.tree import DecisionTreeClassifier
from .base import BaseModel
import numpy as np

class DecisionTreeClassifierModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return self.model.get_params()

    def set_params(self, params: dict):
        self.model.set_params(**params)
