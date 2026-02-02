# Sparsity Analysis

This directory contains tools for analyzing sparsity of activations, weights, and operations in the SpikeYolo hardware simulation.

## Overview

The hardware simulation project analyzes the sparsity characteristics of SpikeYolo model components, including:
- **Weight Sparsity**: Analysis of ternary weight distributions (-1, 0, +1)
- **Activation Sparsity**: Analysis of activation data sparsity (0, 1, 2, 3 values)
- **Operation Sparsity**: Analysis of skipped operations in tree-based vector multiplication units

## Files Description

### Core Analysis Scripts

#### `ProfileSparsity.py`
Main script for profiling sparsity of activations, weights, and operations during convolution.

**Features:**
- Implements tree-based vector multiplication units with different group sizes (2, 4, 8)
- Performs convolution operations and compares with PyTorch reference
- Analyzes skipped operations at different tree levels
- Supports ternary weights (-1, 0, +1) and quantized activations (0, 1, 2, 3)

**Usage:**
```python
python ProfileSparsity.py
```

**Key Functions:**
- `tpu_conv()`: Performs convolution using tree vector multiplication units
- `profile_sparsity()`: Profiles sparsity for different group sizes
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

**Features:**
- Supports skipping operations when input groups are all zeros
- Returns skipped operation counts at each tree level
- Optimized for ternary weight multiplication

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

## Directory Structure

```
hardware_sim/
├── ProfileSparsity.py      # Main sparsity profiling script
├── weight_sparsity.py      # Weight sparsity analysis
├── lif_sparsity.py         # LIF output sparsity analysis
├── VecMulUnit.py           # Tree-based vector multiplication units
├── extract_data/           # Extracted data for analysis
│   └── step13_conv/        # Example convolution layer data
│       ├── input_001.txt
│       ├── input_shape_001.txt
│       ├── weights.txt
│       └── weights_shape.txt
├── weight_sparsity_stats.csv  # Weight statistics output
└── lif_sparsity_stats.csv     # LIF statistics output
```

## Usage Examples

### 1. Profile Operation Sparsity
```python
# Edit ProfileSparsity.py to set your data paths and parameters
in_channels = 256
out_channels = 64
kernel_size = 3
stride = 1
padding = 1
height = 96
width = 160

# Run the analysis
python ProfileSparsity.py
```

### 2. Analyze Weight Sparsity
```python
# Edit weight_sparsity.py to set root_path, expect full extract weight file
root_path = "/path/to/model/directory"
output_csv = "weight_sparsity_stats.csv"

# Run the analysis
python weight_sparsity.py
```

### 3. Analyze LIF Output Sparsity
```python
# Edit lif_sparsity.py to set root_path, expect full extract weight file
root_path = "/path/to/model/directory"
output_csv = "lif_sparsity_stats.csv"

# Run the analysis
python lif_sparsity.py
```

## Key Concepts

### Tree-Based Vector Multiplication
The tree-based approach groups 128 input elements and performs hierarchical multiplication:
- **Group Size 2**: 7 levels (128→64→32→16→8→4→2→1)
- **Group Size 4**: 3 levels (128→32→8→2→1)
- **Group Size 8**: 2 levels (128→16→2→1)

### Operation Skipping
When all elements in a group are zero, the entire group multiplication can be skipped, reducing computation.

### Sparsity Metrics
- **Weight Sparsity**: Percentage of zero weights
- **Activation Sparsity**: Percentage of zero activations
- **Operation Sparsity**: Percentage of skipped operations at each tree level

## Requirements

- Python 3.x
- NumPy
- PyTorch (for reference convolution)
- CSV module (standard library)

## Notes

- Currently supports only `padding=1` and `stride=1` convolutions
- Input channels must be divisible by 128 (for vector multiplication units)
- Weights are assumed to be ternary (-1, 0, +1)
- Activations are quantized to integer values (0, 1, 2, 3)
