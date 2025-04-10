# src/socialgaze/utils/saving_utils.py


import os
import json
from typing import Dict, Any


def save_config_to_json(config_dict: Dict[str, Any], config_path: str) -> None:
    """
    Saves a dictionary to a JSON file.

    Args:
        config_dict (Dict[str, Any]): The config dictionary to save.
        config_path (str): Target path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
