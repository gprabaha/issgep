from .base import BaseModel

class ExponentialForagingModel(BaseModel):
    def __init__(self, lambda_decay=0.1):
        self.lambda_decay = lambda_decay

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass

class OUForagingModel(BaseModel):
    def __init__(self, theta=0.15, sigma=0.2, mu=0.0):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass
