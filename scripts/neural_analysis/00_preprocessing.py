# scripts/neural_analysis/

from socialgaze.config.base_config import BaseConfig
from socialgaze.data.spike_data import SpikeData

if __name__=="__main__":
    # Example usage
    config = BaseConfig()
    spike_data = SpikeData(config)

    # Load from raw if needed
    spike_data.load_from_mat()
    spike_data.save_to_pkl()

    # Load from saved
    df = spike_data.get_data()
