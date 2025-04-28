from .base import BaseModel

class QLearningModel(BaseModel):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass

class SARSAModel(BaseModel):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass
