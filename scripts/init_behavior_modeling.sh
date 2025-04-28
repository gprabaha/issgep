-#!/usr/bin/env bash

# Create folder
mkdir -p src/socialgaze/models/behavior

# Create __init__.py
cat <<EOF > src/socialgaze/models/behavior/__init__.py
"""
Behavioral modeling subpackage: foraging, RL, inverse RL models.
"""

from .base import BaseModel
from .foraging import ExponentialForagingModel, OUForagingModel
from .rl import QLearningModel, SARSAModel
from .inverse_rl import MaxEntIRL

__all__ = [
    "BaseModel",
    "ExponentialForagingModel", "OUForagingModel",
    "QLearningModel", "SARSAModel",
    "MaxEntIRL"
]
EOF

# Create base.py
cat <<EOF > src/socialgaze/models/behavior/base.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for behavior models."""

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass
EOF

# Create foraging.py
cat <<EOF > src/socialgaze/models/behavior/foraging.py
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
EOF

# Create rl.py
cat <<EOF > src/socialgaze/models/behavior/rl.py
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
EOF

# Create inverse_rl.py
cat <<EOF > src/socialgaze/models/behavior/inverse_rl.py
from .base import BaseModel

class MaxEntIRL(BaseModel):
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass
EOF

echo "Behavior modeling templates created successfully!"


