# src/socialgaze/utils/conversion_utils.py

from typing import Any, Dict
from pathlib import Path


def object_to_dict(config_obj: Any) -> Dict[str, Any]:
    """
    Converts a config object to a dictionary suitable for JSON serialization.

    Args:
        config_obj (Any): An object with serializable attributes.

    Returns:
        Dict[str, Any]: Dictionary with all serializable attributes.
    """
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config_obj.__dict__.items()
        if not key.startswith('_')
    }


def assign_dict_attributes_to_object(obj: Any, config_data: Dict[str, Any]) -> None:
    """
    Assigns attributes from a dictionary to a config object, converting path strings to Path objects.

    Args:
        obj (Any): The object to assign attributes to.
        config_data (Dict[str, Any]): The configuration data.
    """
    for key, value in config_data.items():
        if isinstance(value, str) and ("dir" in key or "path" in key or "folder" in key):
            setattr(obj, key, Path(value))
        else:
            setattr(obj, key, value)
