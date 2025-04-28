#!/usr/bin/env bash

# Create folders if they don't exist
mkdir -p scripts
mkdir -p configs/behavior

# Create training script
cat <<EOF > scripts/train_behavior_model.py
import argparse

from socialgaze.models.behavior import ExponentialForagingModel, OUForagingModel, QLearningModel, SARSAModel, MaxEntIRL

def load_data(config):
    # TODO: Implement loading gaze and fixation data
    pass

def train_model(config):
    data = load_data(config)
    model_type = config["model_type"]

    if model_type == "ExponentialForaging":
        model = ExponentialForagingModel()
    elif model_type == "OUForaging":
        model = OUForagingModel()
    elif model_type == "QLearning":
        model = QLearningModel()
    elif model_type == "SARSA":
        model = SARSAModel()
    elif model_type == "MaxEntIRL":
        model = MaxEntIRL()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(data)

    # TODO: Save model or evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/behavior/default.yaml", help="Path to config file")
    args = parser.parse_args()

    # TODO: Load YAML config
    config = {"model_type": "ExponentialForaging"}  # placeholder
    train_model(config)
EOF

# Create default config
cat <<EOF > configs/behavior/default.yaml
# Default config for training behavior models

model_type: ExponentialForaging

# Placeholder for data locations, training hyperparameters
data_path: data/processed/
save_path: outputs/models/behavior/
learning_rate: 0.01
epochs: 50
EOF

echo "Training script and default config created successfully!"
