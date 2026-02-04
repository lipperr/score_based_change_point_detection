# Score-Based Change Point Detection

This repository contains code for numerical experiments and implementations of several online change point detection algorithms. The description of the experiments can be found in the paper "Score-based change point detection via tracking the best of infinitely many experts" by A. Markovich, N. Puchkin available at .

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Datasets](#datasets)
- [Configuration System](#configuration-system)
- [Usage](#usage)
- [Output Formats](#output-formats)
- [Features](#advanced-features)
- [Examples](#examples)

## Overview

This project implements and evaluates multiple online change point detection algorithms:
- **FS** (Score-based): Score-based change point detection via tracking the best of infinitely many experts
- **FLH**: Follow-the-Leading-History
- **FALCON**: Fast online contrastive change point detection
- **KLIEP**: Kullback–Leibler importance estimation procedure
- **M-statistic**: Kernel change point detector with M-statistic

The framework supports two evaluation modes:
- **Stationary mode**: Tune thresholds on stationary data
- **Changepoint mode**: Evaluate algorithms on data with change points, compute metrics (false alarms, detection delays, non-detections)

## Installation

### Prerequisites

- Python 3.8+
- Required packages:
  - `hydra-core` (for configuration management)
  - `omegaconf` (for config handling)
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd score_based_change_point_detection

# Install dependencies (create a virtual environment recommended)
pip install hydra-core omegaconf numpy pandas matplotlib scipy
```

## Project Structure

```
score_based_change_point_detection/
├── algorithms/          # Algorithm implementations
│   ├── fs.py           # Score-based (FS)
│   ├── flh.py          # FLH
│   ├── fast_contrastive_change_point.py  # FALCON
│   ├── kliep.py        # KLIEP
│   ├── m_statistic.py  # M-statistic
│   └── ew.py           # Exponential weighting (shared by FS/FLH)
├── datasets/            # Dataset loaders
│   ├── gaussian.py     # Synthetic Gaussian data
│   ├── wisdm.py        # WISDM activity recognition
│   ├── censrec.py      # CENSREC audio
│   └── occupancy.py    # Room occupancy detection
├── configs/             # Configuration files
│   ├── algorithm/      # Algorithm configurations
│   ├── data/           # Dataset configurations
│   ├── kwargs/         # Parameter grids for experiments
│   ├── run.yaml        # Main config
│   └── threshold_scaling.yaml  # Threshold scaling rules
├── data/                # Raw data files
├── results/             # Output CSV and plots
├── runner.py           # Main runner (single algorithm)
├── performance_runner.py  # Runner for multiple algorithms
├── display_data.py     # Data visualization tool
├── test_functions.py   # Test/evaluation functions
├── run_procedure.py    # Core procedures (tuning, running)
└── utils.py            # Utility functions
```

## Algorithms

### FS (Score-based)
- **Class**: `algorithms.fs.FS`
- **Parameters**:
  - `alpha`: Regularization parameter
  - `ew_params`: Exponential weighting parameters
    - `lambda_`: Regularization
    - `eta`: Learning rate (constant or function)
    - `gamma`: Additional regularization
    - `basis`: Basis function (e.g., `PolyBasis` with `degree`)
    - `xdim`: Input dimension

### FLH (Follow the Leader with History)
- **Class**: `algorithms.flh.FLH`
- **Parameters**: Same as FS

### FALCON
- **Class**: `algorithms.fast_contrastive_change_point.FALCON`
- **Parameters**:
  - `p`: Order parameter
  - `beta`: Regularization parameter
  - `design`: Design type ("multivariate" or "hermite")

### KLIEP
- **Class**: `algorithms.kliep.KLIEP`
- **Parameters**:
  - `window_size`: Window size
  - `sigma`: Kernel bandwidth

### M-statistic
- **Class**: `algorithms.m_statistic.Mstatistic`
- **Parameters**:
  - `window_size`: Window size
  - `sigma`: Kernel bandwidth

## Datasets

### Gaussian (Synthetic)
- **Config**: `configs/data/gaussian_*.yaml`
- **Parameters**:
  - `mu`: Mean
  - `sigma`: Standard deviation
  - `shift`: Change magnitude
  - `shift_type`: "mean" or "std"
  - `dim`: Dimension
  - `size`: Sequence length
  - `cps`: Change point location
  - `runs`: Number of runs
  - `seed`: Random seed

### WISDM
- **Config**: `configs/data/wisdm.yaml`
- **Parameters**:
  - `dim`: Dimension of the data (3)
  - `part`: "val" or "test"
  - `path`: Path to CSV file

### CENSREC
- **Config**: `configs/data/censrec.yaml`
- **Parameters**:
  - `part`: "val" or "test"
  - `data_part`: clean recording, SNR15, SNR20
  - `path`: Path to audio files

### Occupancy
- **Config**: `configs/data/occupancy.yaml`
- **Parameters**:
  - `dim`: Dimension
  - `part`: "val" or "test"
  - `path`: Path to JSON file

## Configuration System

The project uses Hydra for configuration management. Configurations are organized hierarchically:

### Main Config (`configs/run.yaml`)

```yaml
defaults:
  - data: "wisdm"           # Dataset config
  - algorithm: fs            # Algorithm config
  - kwargs: wisdm             # Parameter grid config
  - _self_

runner:
  mode: "stationary"         # or "changepoint"
  seed: 1
  save_path: "results/"
  dataset_name: "wisdm"      # Used for filenames
  threshold_scaling_path: "configs/threshold_scaling.yaml"
  # thresholds_csv_path: "results/stationary_fs_wisdm_ths.csv"  # Optional: precomputed thresholds
```

### Algorithm Configs (`configs/algorithm/`)

Example: `configs/algorithm/fs.yaml`
```yaml
_target_: algorithms.fs.FS
alpha: 1e-6
ew_params:
  xdim: ${data.dim}
  lambda_: 1.5
  eta: 0.2
  gamma: 1e-6
  basis:
    _target_: algorithms.ew.PolyBasis
    degree: 2
    xdim: ${data.dim}
```

### Dataset Configs (`configs/data/`)

Example: `configs/data/wisdm.yaml`
```yaml
_target_: datasets.WISDM
dim: 3
part: "val"
path: "data/WISDM/sample_0.csv"
save_path: ${runner.save_path}
```

### Parameter Grid Configs (`configs/kwargs/`)

Define parameter lists to test. Example: `configs/kwargs/exp1.yaml`
```yaml
fs:
  alpha_list: [1e-3]
  eta_list: [0.1]
  lambda_list: [0.05]

flh:
  alpha_list: [0.05]
  eta_list: [0.2]
  lambda_list: [0.5]

falcon:
  p_list: [2]
  beta_list: [0.3, 0.5, 0.7, 0.9, 1.1, 1.5]

kliep:
  sigma_list: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2]

mstat:
  sigma_list: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3]
```

## Usage

### Basic Usage

Run a single algorithm on a dataset:

```bash
python runner.py
```

This uses defaults from `configs/run.yaml`.

### Override Configuration

Override any config parameter via command line:

```bash
# Change dataset
python runner.py data=wisdm

# Change algorithm
python runner.py algorithm=flh

# Change parameter grid
python runner.py kwargs=exp2

# Change mode
python runner.py runner.mode=changepoint

# Change dataset name (for filenames)
python runner.py runner.dataset_name=wisdm_test

# Change save path
python runner.py runner.save_path="myfolder/"

# Override dataset parameters
python runner.py data=occupancy data.part=test

# Override algorithm parameters
python runner.py algorithm=fs algorithm.alpha=1e-4
```

### Complete Example

```bash
python runner.py \
  data=occupancy \
  kwargs=occupancy \
  runner.save_path="results/" \
  runner.dataset_name="occupancy_test" \
  data.part="test" \
  runner.mode="changepoint" \
  algorithm=mstat
```

### Run Multiple Algorithms

Use `performance_runner.py` to compare multiple algorithms on a single change point time series:

```bash
python performance_runner.py
```

This uses `configs/run_for_plot.yaml` which defines multiple algorithm instances.

### Display Dataset

Visualize a dataset:

```bash
python display_data.py data=wisdm runner.dataset_name=wisdm
```

## Output Formats

### Stationary Mode

Output CSV: `{mode}_{algorithm}_{dataset_name}_ths.csv`

Columns:
- Algorithm-specific parameters (e.g., `alpha`, `lambda`, `eta`, `gamma` for FS/FLH)
- `threshold`: Tuned threshold value

### Changepoint Mode

Output CSV: `{mode}_{algorithm}_{dataset_name}_ths.csv`

Columns:
- Algorithm-specific parameters
- `threshold`: Threshold used
- `FA`: False alarms count
- `DD`: Detection delay (format: `mean±std`)
- `ND`: Non-detections count

Output Plot: `{mode}_{algorithm}_{dataset_name}.png`
- Shows test statistic over time with detected change points
- Vertical lines indicate true change points

##  Features

### Threshold Scaling

The framework supports dataset/algorithm-specific threshold scaling via `configs/threshold_scaling.yaml`. Example:

```yaml
wisdm:
  fs:   { add: 0.1 } 
occupancy:
  kliep: { factor: 2}
```

**Scaling rules**:
- `factor`: Multiply threshold by this value
- `add`: Add this value to threshold

Scaling is applied automatically based on `dataset.name`, it works regardless of the filename you choose.

### Precomputed Thresholds

To reuse thresholds from a previous stationary run:

```yaml
runner:
  thresholds_csv_path: "results/stationary_fs_wisdm_ths.csv"
```

The system will:
1. Load thresholds from CSV matching parameter combinations
2. Use precomputed thresholds when available
3. Compute new thresholds only for missing parameter combinations

### Custom Output Names

Override default filenames:

```yaml
runner:
  output_names:
    csv: "custom_results.csv"
    plot: "custom_plot.png"
```

## Examples

### Example 1: Tune Thresholds (Stationary Mode)

```bash
python runner.py \
  data=wisdm \
  algorithm=fs \
  kwargs=wisdm \
  runner.mode=stationary \
  runner.save_path=results/
```

### Example 2: Evaluate on Change Points

```bash
python runner.py \
  data=wisdm \
  algorithm=fs \
  kwargs=wisdm \
  runner.mode=changepoint \
  runner.save_path=results/ \
  +runner.thresholds_csv_path=results/stationary_fs_wisdm_ths.csv
```

### Example 3: Test Multiple Parameter Combinations

```bash
python runner.py \
  data=gaussian_1dmean \
  algorithm=fs \
  kwargs=exp1 \
  runner.mode=changepoint \
  runner.dataset_name=gaussian_1dmean
```

### Example 4: Compare a set of algorithms on a time series with a single change point and plot the results

```bash
python performance_runner.py \
  data=gaussian_3dmean \
  data.runs=1 \
  runner.mode=changepoint
```

## Notes

- **Hydra outputs**: Hydra creates an `outputs/` directory with timestamped runs. To disable this, set `HYDRA_FULL_ERROR=1` or configure Hydra output behavior.
- **Dataset names**: `runner.dataset_name` is used for filenames only. Threshold scaling uses `dataset.name` (the dataset class attribute).
- **Parameter grids**: The kwargs configs define parameter lists. All combinations are tested in a grid search.
- **Sequential vs Independent**: Datasets can be `sequential` (single stream) or `independent` (multiple runs). The evaluation procedure adapts accordingly.

## Citation

If you use this code, please cite:
```bibtex

```


## References (algorithm papers)

**Score-based (FS)** — this repository’s paper (see Citation above).

- **FLH**: Follow-the-Leading-History
```bibtex
@article{hazan07b,
    title = {Adaptive Algorithms for Online Decision Problems},
    author = {Elad Hazan and C. Seshadhri},
    journal = {Electronic Colloquium on Computational Complexity, Technical Report},
    year = {2007},
    volume = {07-088}
}
```
- **FALCON**: Fast online contrastive change point detection
```bibtex
@misc{goldman23,
      title = {A Contrastive Approach to Online Change Point Detection}, 
      author = {Artur Goldman and Nikita Puchkin and Valeriia Shcherbakova and Uliana Vinogradova},
      year = {2023},
      howpublished = {Preprint. ArXiv:2206.10143} 
}
```
- **KLIEP**: Kullback–Leibler importance estimation procedure
```bibtex
@article{sugiyama08,
  author  = {Sugiyama, Masashi and Suzuki, Taiji and Nakajima, Shinichi and Kashima, Hisashi and von B\"{u}nau, Paul and Kawanabe, Motoaki},
  title   = {Direct importance estimation for covariate shift adaptation},
  journal = {Annals of the Institute of Statistical Mathematics},
  volume  = {60},
  year    = {2008},
  number  = {4},
  pages   = {699--746}
}
```
- **M-statistic**: Kernel change point detector with M-statistic
```bibtex
@inproceedings{li15,
    author = {Li, Shuang and Xie, Yao and Dai, Hanjun and Song, Le},
    booktitle = {Advances in Neural Information Processing Systems},
    pages = {},
    title = {M-{S}tatistic for {K}ernel {C}hange-{P}oint {D}etection},
    volume = {28},
    year = {2015}
}
```