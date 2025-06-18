# Social Gaze Analysis

A comprehensive Python package for analyzing eye-tracking and neural data in social gaze experiments. This repository provides tools for processing behavioral data (fixations, saccades, gaze patterns) and neural recordings from multi-agent social interaction studies.

## Overview

This package is designed to process and analyze:
- **Eye-tracking data** from dual-agent social interaction experiments
- **Neural recordings** synchronized with behavioral data
- **Cross-correlation analysis** between agents' gaze patterns
- **Fixation detection** and region-of-interest (ROI) classification
- **Principal component analysis** of neural activity during social interactions

## Repository Structure

```
├── data/
│   ├── processed/           # Processed datasets and analysis outputs
│   │   ├── binary_vectors/  # Binary time series for different gaze behaviors
│   │   └── ...
│   └── raw/                 # Raw data files (not tracked)
├── jobs/                    # HPC job submission scripts and logs
├── notebooks/               # Jupyter notebooks for analysis
├── outputs/                 # Analysis results and figures
│   └── crosscorr/          # Cross-correlation analysis results
├── scripts/                 # Analysis pipeline scripts
│   ├── behav_analysis/     # Behavioral data processing
│   ├── neural_analysis/    # Neural data processing
│   ├── modeling/           # Statistical modeling
│   └── visualization/      # Plotting and visualization
└── src/socialgaze/         # Main package source code
    ├── config/             # Configuration management
    ├── data/               # Data loading and handling
    ├── features/           # Feature extraction algorithms
    ├── utils/              # Utility functions
    └── visualization/      # Plotting modules
```

## Installation

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd socialgaze
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate gaze_processing
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Requirements

Key dependencies include:
- Python 3.10+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib (for parallel processing)

See `requirements.txt` for the complete list.

## Usage

### Configuration

The package uses a hierarchical configuration system:
- `BaseConfig`: Core paths and environment detection
- `FixationConfig`: Fixation detection parameters
- Additional configs for specific analyses

Configurations automatically detect whether you're running on:
- Local machine
- HPC cluster (Grace/Milgram)
- Specific user environments

### Basic Analysis Pipeline

#### 1. Fixation Detection

Process eye-tracking data to identify fixations and saccades:

```bash
# Single session processing
python scripts/behav_analysis/01_fixation_detection.py --session 20240101 --run 1 --agent m1

# Full dataset processing (HPC)
python scripts/behav_analysis/01_fixation_detection.py
```

#### 2. Interactivity Detection

Identify periods of social interaction:

```bash
python scripts/behav_analysis/02_interactivity_detection.py
```

#### 3. Cross-correlation Analysis

Analyze temporal relationships between agents' gaze patterns:

```bash
python scripts/behav_analysis/04_inter_agent_crosscorr.py
```

#### 4. Neural Analysis

Extract peri-stimulus time histograms (PSTHs) and perform dimensionality reduction:

```bash
python scripts/neural_analysis/01_psth_extraction.py
python scripts/neural_analysis/02_pc_projection.py
```

### Key Features

#### Fixation Detection
- Velocity-based fixation detection algorithm
- ROI classification (face vs. out-of-ROI)
- Binary vector generation for time series analysis

#### Cross-correlation Analysis
- Inter-agent gaze synchrony measurement
- Shuffled controls for statistical validation
- Multiple behavioral event types (fixations, saccades)

#### Neural Analysis
- PSTH extraction around behavioral events
- PCA-based dimensionality reduction
- Trajectory analysis in neural state space

#### HPC Integration
- Automatic job array generation
- SLURM integration for parallel processing
- Environment-specific resource allocation

## Data Structure

### Input Data
- **Eye-tracking**: Position data, pupil diameter, timestamps
- **Neural**: Spike times, channel information
- **ROI**: Region-of-interest vertex coordinates
- **Behavioral**: Event timestamps and classifications

### Output Data
- **Fixations/Saccades**: Detected events with ROI labels
- **Binary Vectors**: Time series for different behavioral states
- **Cross-correlations**: Inter-agent synchrony measures
- **Neural Features**: PSTHs and principal component trajectories

## Configuration Examples

### Fixation Detection Configuration

```python
from socialgaze.config.fixation_config import FixationConfig

config = FixationConfig()
config.detect_fixations_again = True
config.update_labels_in_dfs = True
config.binary_vector_types_to_generate = [
    "face_fixation",
    "saccade_to_face",
    "saccade_from_face"
]
```

### HPC Job Submission

The package automatically generates job arrays for HPC processing:

```bash
# Generates and submits fixation detection jobs
python scripts/behav_analysis/01_fixation_detection.py
# This creates job arrays and submits via SLURM
```

## Development

### Package Structure

- **config/**: Configuration classes with automatic environment detection
- **data/**: Data loading and preprocessing utilities
- **features/**: Core analysis algorithms (fixation detection, cross-correlation, etc.)
- **utils/**: Helper functions for file I/O, path management, HPC utilities
- **visualization/**: Plotting functions and result visualization

### Adding New Analyses

1. Create analysis script in appropriate `scripts/` subdirectory
2. Add configuration class in `src/socialgaze/config/`
3. Implement core algorithms in `src/socialgaze/features/`
4. Add visualization functions if needed

### Testing

```bash
# Run tests
python -m pytest tests/
```

## HPC Usage

The package is designed for both local and HPC environments:

### Grace Cluster
- Automatic detection of Grace environment
- Optimized resource allocation
- Environment: `gaze_otnal`

### Milgram Cluster
- Milgram-specific configurations
- Environment: `gaze_processing`

### Job Management
- Automatic SLURM script generation
- Resource optimization based on data size
- Logging and error handling

## Citation

If you use this package in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## Contact

[Contact information to be added]

---

**Note**: This package is actively developed for social neuroscience research. For questions about specific analyses or feature requests, please open an issue on GitHub.