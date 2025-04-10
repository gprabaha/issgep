# src/socialgaze/utils/config_utils.py


import os
import logging
import sys

from socialgaze.config.base_config import BaseConfig

logger = logging.getLogger(__name__)

def ensure_config_exists(config_path: str):
    if not os.path.exists(config_path):
        logger.warning("Config file not found at: %s", config_path)
        logger.info("Attempting to generate config automatically...")
        try:
            config = BaseConfig()
        except subprocess.CalledProcessError:
            logger.error("Failed to generate config. Exiting.")
            sys.exit(1)