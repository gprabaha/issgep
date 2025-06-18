
# Interactive States during Social Gaze Exchange in Primates (issgep)

This repository contains the data processing and analysis pipeline used to study dynamic social gaze behavior and associated neural activity in primates. It includes tools to detect behavioral events (e.g., fixations, saccades), extract neural responses aligned to these events, compute inter-agent behavioral metrics, and analyze population neural dynamics.

---

## 🧠 Project Scope

We focus on understanding how social gaze behavior unfolds over time between interacting agents and how neural signals track these interactions. Our pipeline allows:

- Detection of fixations and saccades
- Classification of gaze events toward/away from faces or objects
- Extraction of peri-event neural responses (PSTHs)
- Dimensionality reduction (PCA) of neural population activity
- Calculation of cross-correlations and fixation probabilities between agents

---

## 🚀 Getting Started

### 1. Clone and Set Up the Environment

```bash
git clone git@github.com:gprabaha/issgep.git
cd issgep
conda env create -f environment.yml
conda activate issgep
```

### 2. Example: Detect Fixations

```bash
python scripts/behav_analysis/01_fixation_detection.py --session 20230718 --run 1 --agent m1
```

---

## 🗂️ Organization Overview

The repository follows a modular structure:

* `src/socialgaze/`: Core library (functions, classes, utils)
* `scripts/`: Analysis scripts organized by topic (e.g., behavior, neural, modeling)
* `jobs/`: SLURM job scripts and array generators for HPC processing
* `data/`: Raw and processed data files
* `outputs/`: Saved results (e.g., cross-correlations, PCA results)

---

## 🧩 How to Add New Functionality

### ✅ Functions

* Place utility functions in `src/socialgaze/utils/` when they are general-purpose.
* Use clear, testable inputs and outputs — avoid using or modifying global state.
* If function logic is tightly coupled to a module (e.g., fixations), place it in the relevant feature file (e.g., `fixation_utils.py`).

```python
# Example: utils/fixation_utils.py
def get_fixation_duration(fixation):
    return fixation["end_time"] - fixation["start_time"]
```

### ✅ Classes

* Add new classes to `src/socialgaze/features/` or `src/socialgaze/data/` depending on their role.
* Inherit from `BaseConfig` for config-aware tools, or from a relevant parent class like `FixationDetector`.
* Each class should have:

  * `__init__` to load config and inputs
  * `run()` or `execute()` method to launch its logic
  * Optional `save()` and `load()` methods if state needs to be persisted

```python
class NewAnalysisTool(BaseConfig):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.results = None

    def run(self):
        self.results = ...  # Perform core logic

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.results, f)
```

### ✅ Configs

* All config classes live in `src/socialgaze/config/` and inherit from `BaseConfig`.
* Create a new file like `new_feature_config.py` and define your config class there.
* Store default values and control flags in the constructor.

```python
class NewFeatureConfig(BaseConfig):
    def __init__(self, config_path=None):
        super().__init__(config_path)
        self.enable_caching = True
        self.window_size = 300
```

---

## 🧪 Writing Analysis Scripts

* Scripts go into the relevant subfolder of `scripts/`

  * Behavioral analysis → `scripts/behav_analysis/`
  * Neural analysis → `scripts/neural_analysis/`
  * Modeling → `scripts/modeling/`
* Scripts should:

  * Load the appropriate config and data
  * Call the appropriate feature class or function
  * Save output in `data/processed/` or `outputs/`
  * Be executable via command-line arguments using `argparse`

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str)
    parser.add_argument("--run", type=int)
    args = parser.parse_args()

    config = SomeConfig()
    tool = SomeFeatureClass(config)
    tool.run(session=args.session, run=args.run)
```

---

## 📊 Outputs and Reproducibility

* All major outputs are stored under `data/processed/` and `outputs/`
* Binary vectors, spike data, and fixation events are serialized using `pickle`
* Intermediate outputs (e.g., job-wise temp files) are stored in `data/processed/temp/`
* To rerun any step, delete its corresponding output file or set `remake=True` in config

---

## ⚙️ HPC Execution

To run jobs on SLURM, use the job scripts in `jobs/scripts/`. Each analysis stage has a paired job script and job array file. Example:

```bash
sbatch jobs/scripts/run_fixation_job.sh
```

