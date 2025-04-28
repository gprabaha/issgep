from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for behavior models."""

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass
