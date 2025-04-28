from .base import BaseModel

class MaxEntIRL(BaseModel):
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass
