# src/socialgaze/config/preprocessing_params.py

from socialgaze.config.base_config import BaseConfig

def get_preprocessing_config(base_cfg: BaseConfig) -> dict:
    return {
        "resample_rate": 1000,
        "lowpass_cutoff": 30,
        "nan_padding_window": 5,
        "save_dir": base_cfg.output_dir / "preprocessing"
    }
