# Sparsity Analysis

This directory contains tools for analyzing sparsity of activations, weights, and operations in the SpikeYolo hardware simulation.

## Overview

The hardware simulation project analyzes the sparsity characteristics of SpikeYolo model components, including:
- **Weight Sparsity**: Analysis of ternary weight distributions (-1, 0, +1)
- **Activation Sparsity**: Analysis of activation data sparsity (0, 1, 2, 3 values)
- **Operation Sparsity**: Analysis of skipped operations in tree-based vector multiplication units

## Files Description

### Core Analysis Scripts

#### `ProfileModelSparsity.py`
Main script for profiling sparsity across all convolution layers in a model.

**Features:**
- Automatically finds all convolution layers in the model directory
- Filters layers with input channels >= 128 and divisible by 128
- Randomly selects one output channel for each layer
- Performs convolution using tree-based vector multiplication units
- Compares results with PyTorch reference (strict equality check)
- Records input sparsity, weight sparsity, and operation sparsity at each tree level
- Exports comprehensive results to CSV with average statistics

**Usage:**
```python
from ProfileModelSparsity import profile_model_sparsity

base_path = './refs/inference_data_TWN_bn_fused/model'
output_csv = './sparsity_profile_results.csv'
profile_model_sparsity(base_path, output_csv)
```

**Output CSV Columns:**
- Layer name (relative path from base_path)
- In channels, kernel_size, index_channel
- input_sparsity, weight_sparsity
- Level(128->32) skip counts, Level(128->32) skip rate
- Level(32->16) skip counts, Level(32->16) skip rate
- ... (for all 6 levels)
- compute count, weights_update_time
- Average row with aggregated statistics

#### `ProfileLayerSparsity.py`
Script for profiling sparsity of a single convolution layer with different tree configurations.

**Features:**
- Tests different tree group sizes (2, 4, 8)
- Performs convolution operations and compares with PyTorch reference
- Analyzes skipped operations at different tree levels
- Supports ternary weights (-1, 0, +1) and quantized activations (0, 1, 2, 3)

**Usage:**
```python
python ProfileLayerSparsity.py
```

**Key Functions:**
- `profile_layer_sparsity()`: Profiles sparsity for different group sizes
- `compare_result()`: Compares TPU results with PyTorch reference

#### `weight_sparsity.py`
Analyzes weight sparsity statistics across all layers of the model.

**Features:**
- Recursively searches for weight files in the model directory
- Calculates sparsity statistics (zero count, +1 count, -1 count)
- Exports results to CSV file (`weight_sparsity_stats.csv`)

**Usage:**
```python
python weight_sparsity.py
```

**Output:**
- CSV file with per-layer weight statistics
- Summary statistics including total parameters and sparsity rates

#### `lif_sparsity.py`
Analyzes LIF (Leaky Integrate-and-Fire) neuron output sparsity.

**Features:**
- Loads LIF output files (integer values: 0, 1, 2, 3)
- Calculates distribution of different output values
- Exports statistics to CSV file (`lif_sparsity_stats.csv`)

**Usage:**
```python
python lif_sparsity.py
```

### Components

#### `VecMulUnit.py`
Implements tree-based vector multiplication units for hardware simulation.

**Classes:**
- `naive_vec_mul_128`: Naive vector multiplication (baseline)
- `tree_vec_mul_128_group2`: Tree structure with group size 2 (128→64→32→16→8→4→2→1)
- `tree_vec_mul_128_group4`: Tree structure with group size 4 (128→32→8→2→1)
- `tree_vec_mul_128_group8`: Tree structure with group size 8 (128→16→2→1)
- `tree_vec_mul_128_group4_adder2`: Tree structure with group size 4 and 2-input adders (128→32→16→8→4→2→1)

**Features:**
- **Smart Operation Skipping**: 
  - Level 1 (multiplication): Skips when input group is all zeros **OR** weight group is all zeros
  - Other levels (addition): Skips when both inputs are zero
- Returns skipped operation counts at each tree level
- Optimized for ternary weight multiplication (-1, 0, +1)

#### `VecMulGroup4Adder2.py`
Alternative implementation of `tree_vec_mul_128_group4_adder2` with the same smart skipping logic.

#### `utils.py`
Utility functions for convolution operations and sparsity analysis.

**Key Functions:**
- `do_conv_torch()`: PyTorch reference convolution
- `tpu_conv_first_channel()`: TPU convolution for one randomly selected output channel
- `load_data()`: Load data from text files
- `compare_result()`: Compare TPU and PyTorch results
- `data_sparsity()`: Calculate input data sparsity
- `weights_sparsity()`: Calculate weight sparsity

## Data Format

### Input Data Format
- **Weights**: Text files containing ternary weights (-1, 0, +1)
  - `weights.txt`: Flattened weight values
  - `weights_shape.txt`: Shape information [out_channels, in_channels, kernel_h, kernel_w]

- **Activations**: Text files containing quantized activations (0, 1, 2, 3)
  - `input_001.txt`: Flattened input values
  - `input_shape_001.txt`: Shape information [in_channels, height, width]

### Output Data Format
- **CSV Statistics**: 
  - `weight_sparsity_stats.csv`: Per-layer weight statistics
  - `lif_sparsity_stats.csv`: Per-layer LIF output statistics
  - `sparsity_profile_results.csv`: Comprehensive sparsity analysis for all convolution layers
    - Includes input sparsity, weight sparsity, skip counts and rates for each tree level
    - Contains an average row with aggregated statistics

## Directory Structure

```
hardware_sim/
├── ProfileModelSparsity.py  # Model-wide sparsity profiling (main script)
├── ProfileLayerSparsity.py  # Single layer sparsity profiling
├── weight_sparsity.py       # Weight sparsity analysis
├── lif_sparsity.py          # LIF output sparsity analysis
├── VecMulUnit.py            # Tree-based vector multiplication units
├── utils.py                 # Utility functions
├── extract_data/            # Extracted data for analysis
│   └── step13_conv/         # Example convolution layer data
│       ├── input_001.txt
│       ├── input_shape_001.txt
│       ├── weights.txt
│       └── weights_shape.txt
├── weight_sparsity_stats.csv     # Weight statistics output
├── lif_sparsity_stats.csv        # LIF statistics output
└── sparsity_profile_results.csv  # Comprehensive sparsity profile results
```

## Usage Examples

### 1. Analyze Weight Sparsity
```python
# if weights are stored as [-a, 0, a], edit load_weights:
zero_count = np.sum(weights == 0)
one_count = np.sum(weights > 0)
negative_one_count = np.sum(weights < 0)


# Edit weight_sparsity.py to set root_path, expect full extract weight file
root_path = "/path/to/model/directory"
output_csv = "weight_sparsity_stats.csv"

# Run the analysis
python weight_sparsity.py
```

### 2. Analyze LIF Output Sparsity
```python
# Edit lif_sparsity.py to set root_path, expect full extract weight file
root_path = "/path/to/model/directory"
output_csv = "lif_sparsity_stats.csv"

# Run the analysis
python lif_sparsity.py
```

### 3. Profile Model-Wide Sparsity
```python
from ProfileModelSparsity import profile_model_sparsity

# Set the model data path
base_path = './refs/inference_data_TWN_bn_fused/model'
output_csv = './sparsity_profile_results.csv'

# Run the analysis
profile_model_sparsity(base_path, output_csv)
```

This will:
- Find all convolution layers with input channels >= 128 and divisible by 128
- For each layer, randomly select one output channel
- Calculate input sparsity and weight sparsity
- Perform convolution using tree-based units and record skip statistics
- Export results to CSV with average statistics

### 4. Profile Single Layer Sparsity
```python
# Edit ProfileLayerSparsity.py to set your data paths and parameters
in_channels = 256
out_channels = 64
kernel_size = 3
stride = 1
padding = 1
height = 96
width = 160

# if weights are stored as [-a, 0, a], uncomment the code
# weights = np.where(weights > 0, 1, np.where(weights < 0, -1, 0))

# Run the analysis
python ProfileLayerSparsity.py

# example output:
'''
=============================profile sparsity start======================================
=============================analyze the sparsity of weights start======================================
weights shape: (64, 256, 3, 3)
sparsity of weights: 0.5226372612847222
=============================analyze the sparsity of weights done======================================


=============================analyze the sparsity of data start======================================
data shape: (256, 96, 160)
sparsity of data: 0.8386604309082031
equal to 1 of data: 0.1211212158203125
equal to 2 of data: 0.02939580281575521
equal to 3 of data: 0.010822550455729166
=============================analyze the sparsity of data done======================================


=============================group size = 8======================================
============compare with reference result=============
group size = 8, max_diff: 0.0, mean_diff: 0.0, max_relative_diff: 0.0
skip_recordings: [1202271.   23737.]
total_do_compute_count: 276480
weights_update_time: 18
level1 (128 -> 16)    skipped operation rate: 0.2717807345920139
level2 (16 -> 2)      skipped operation rate: 0.0429271556712963


=============================group size = 4======================================
============compare with reference result=============
group size = 4, max_diff: 0.0, mean_diff: 0.0, max_relative_diff: 0.0
skip_recordings: [4939490.  726536.   27773.]
total_do_compute_count: 276480
weights_update_time: 18
level1 (128 -> 32)    skipped operation rate: 0.5583010073061343
level2 (32 -> 8)      skipped operation rate: 0.32847583912037037
level3 (8 -> 2)       skipped operation rate: 0.05022605613425926


=============================group size = 2======================================
============compare with reference result=============
group size = 2, max_diff: 0.0, mean_diff: 0.0, max_relative_diff: 0.0
skip_recordings: [14114491.  6557511.  2498672.   768944.   187755.    41892.]
total_do_compute_count: 276480
weights_update_time: 18
level1 (128 -> 64)    skipped operation rate: 0.7976668181242766
level2 (64 -> 32)     skipped operation rate: 0.7411827935112847
level3 (32 -> 16)     skipped operation rate: 0.5648401331018519
level4 (16 -> 8)      skipped operation rate: 0.3476490162037037
level5 (8 -> 4)       skipped operation rate: 0.1697726779513889
level6 (4 -> 2)       skipped operation rate: 0.07575954861111112


=============================profile sparsity done======================================
'''
```

## Key Concepts

### Tree-Based Vector Multiplication
The tree-based approach groups 128 input elements and performs hierarchical multiplication:
- **Group Size 2**: 7 levels (128→64→32→16→8→4→2→1)
- **Group Size 4**: 3 levels (128→32→8→2→1)
- **Group Size 8**: 2 levels (128→16→2→1)
- **Group Size 4 with 2-input Adders**: 6 levels (128→32→16→8→4→2→1)

### Operation Skipping
The implementation uses smart skipping logic to reduce computation:

**Level 1 (Multiplication):**
- Skips when **input group is all zeros** OR **weight group is all zeros**
- This is because: `0 × weight = 0` and `input × 0 = 0`

**Other Levels (Addition):**
- Skips when **both inputs are zero**
- This is because: `0 + 0 = 0`

This dual-condition skipping significantly improves sparsity utilization compared to only checking input zeros.

### Sparsity Metrics
- **Weight Sparsity**: Percentage of zero weights
- **Activation Sparsity**: Percentage of zero activations
- **Operation Sparsity**: Percentage of skipped operations at each tree level

## Requirements

- Python 3.x
- NumPy
- PyTorch (for reference convolution)
- Pandas (for data manipulation and CSV export)
- CSV module (standard library)

## Notes

- **ProfileModelSparsity.py**: 
  - Only processes layers with input channels >= 128 and divisible by 128
  - Randomly selects one output channel per layer for analysis
  - Performs strict equality check between TPU and PyTorch results
  - Supports various padding and stride values (read from layer parameter files)

- **ProfileLayerSparsity.py**:
  - Currently supports only `padding=1` and `stride=1` convolutions
  - Tests multiple tree configurations (group sizes 2, 4, 8)

- **General**:
  - Input channels must be divisible by 128 (for vector multiplication units)
  - Weights are assumed to be ternary (-1, 0, +1)
  - Activations are quantized to integer values (0, 1, 2, 3)
  - All results are verified against PyTorch reference for correctness
