from .base import BaseConfig
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BehaviorModelConfig(BaseConfig):
    model_type: str = "ExponentialForaging"
    
    # Paths
    data_path: str = "data/processed/"
    save_path: str = "outputs/models/behavior/"

    # Training Hyperparameters
    lambda_decay: Optional[float] = 0.1  # For ExponentialForaging
    learning_rate: Optional[float] = 0.01  # For MaxEntIRL
    epochs: int = 50

    # Optional Preprocessing flags
    use_fixations_only: bool = True
    include_saccades: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Add any validation if needed here
