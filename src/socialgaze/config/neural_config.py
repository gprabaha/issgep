# src/socialgaze/config/neural_params.py

from socialgaze.config.base_config import BaseConfig

def get_neural_config(base_cfg: BaseConfig) -> dict:
    return {
        "spike_align_window": [-0.5, 1.0],
        "roi_filter": None,
        "spike_threshold": 5.0,
        "save_dir": base_cfg.output_dir / "neural_data"
    }
