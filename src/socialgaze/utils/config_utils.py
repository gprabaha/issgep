# src/socialgaze/utils/config_utils.py


import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def ensure_config_exists(config_path: str):
    if not os.path.exists(config_path):
        logger.warning("Config file not found at: %s", config_path)
        logger.info("Attempting to generate config automatically...")
        try:
            subprocess.run(["python", "scripts/setup/make_config_file.py"], check=True)
            logger.info("Config generated at %s", config_path)
        except subprocess.CalledProcessError:
            logger.error("Failed to generate config. Exiting.")
            sys.exit(1)