# scripts/setup/make_config_file.py

import os
from socialgaze.config.base_config import BaseConfig

def main():
    config = BaseConfig()

    save_path = "config/saved_configs/milgram_default.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    config.save_to_file(save_path)
    print(f"Config saved to {save_path}")


if __name__ == "__main__":
    main()
