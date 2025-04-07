# config/analysis_params.py
from config.base_config import BaseConfig

def get_analysis_config(base_cfg: BaseConfig) -> dict:
    return {
        "neural_bin_size": 0.05,
        "smoothing_kernel": "gaussian",
        "aligned_to": "fixation_onset",
        "save_dir": base_cfg.output_dir / "analysis_results"
    }
