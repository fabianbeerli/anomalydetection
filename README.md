# Unsupervised Anomaly Detection in S&P 500: A Comparative Approach

## Project Overview
This project implements a comparative analysis of anomaly detection algorithms (AIDA, Isolation Forest, and Local Outlier Factor) on S&P 500 market data.

## Prerequisites
- Python 3.8+
- C++ Compiler (g++)
- OpenMP support (for AIDA)
- Homebrew (for macOS users to install OpenMP)
- Virtual environment recommended

## Setup and Installation

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. macOS Users: Install OpenMP Support (required for AIDA)
```bash
brew install libomp
```

## Workflow Steps

### Step 1: Data Retrieval
**Script:** `scripts/retrieve_data.py`  
**Purpose:** Download S&P 500 index and constituent stock data  
**Output:** Raw CSV files in `data/raw/` directory
```bash
python scripts/retrieve_data.py
```

### Step 2: Data Preprocessing
**Script:** `scripts/prepare_data.py`  
**Purpose:** Process raw data, engineer features, create subsequence datasets  
**Output:** Processed CSV files in `data/processed/` directory
```bash
python scripts/prepare_data.py
```

### Step 3: Anomaly Detection

#### Run All Algorithms
**Script:** `scripts/run_all_algorithms.py`  
**Purpose:** Run all anomaly detection algorithms (AIDA, Isolation Forest, LOF, and temporal variants)  
**Output:** Results for all algorithms in `data/algorithm_results/`
```bash
python scripts/run_all_algorithms.py
```

Parameters:
- `--data`: Path to the processed S&P 500 CSV file
- `--output`: Directory to save algorithm results
- `--algorithms`: Algorithms to run (options: aida, iforest, lof, temporal)
- `--window-size`: Window size for temporal algorithms
- `--step`: Step size for temporal algorithms

Example with custom parameters:
```bash
python scripts/run_all_algorithms.py --data data/processed/index_GSPC_processed.csv --output data/custom_results --algorithms iforest lof --window-size 10 --step 2
```

### Step 4: Process and Analyze Results
**Script:** `scripts/process_results.py`  
**Purpose:** Analyze and visualize results from different algorithms  
**Output:** Comparative analysis and visualizations in `data/analysis_results/`
```bash
python scripts/process_results.py
```

Parameters:
- `--results`: Directory containing algorithm results
- `--output`: Directory to save analysis results
- `--data`: Path to the original processed data file (for time index)
- `--window-size`: Window size used in temporal analysis
- `--step`: Step size used in temporal analysis

### Step 5: Time Series Analysis
Temporal analysis is supported for both Isolation Forest and LOF algorithms. This allows for sliding window anomaly detection, which is particularly useful for identifying anomalies that occur over sequences of trading days.

```bash
# Run temporal analysis with default window size (5) and step (1)
python scripts/run_all_algorithms.py --algorithms temporal

# Run with custom window size and step
python scripts/run_all_algorithms.py --algorithms temporal --window-size 3 --step 1
```

The temporal analysis creates overlapping subsequences (e.g., days 1,2,3 compared to 2,3,4, etc.) which helps identify anomalies in the time series patterns rather than just individual data points.

## About the Algorithms

### AIDA (Analytic Isolation and Distance-based Anomaly)
AIDA is a parameter-free unsupervised anomaly detection algorithm that combines metrics of distance with isolation principles. The implementation used is based on the original C++ code by Souto Arias et al. (2023). AIDA offers robust performance across diverse data distributions and provides the TIX (Tempered Isolation-based eXplanation) framework for explaining anomalies.

### Isolation Forest
Isolation Forest is an unsupervised algorithm based on the principle that anomalies are "few and different" and therefore easier to isolate than normal points. It constructs isolation trees and measures the path length required to isolate each data point, with shorter paths indicating likely anomalies.

### Local Outlier Factor (LOF)
LOF is a density-based anomaly detection algorithm that compares the local density of a point to the local densities of its neighbors. Points with substantially lower density than their neighbors receive a higher LOF score, indicating they are more likely to be anomalies.

## Project Structure
- `data/`
  - `raw/`: Original downloaded data
  - `processed/`: Preprocessed and feature-engineered data
  - `algorithm_results/`: Results from running anomaly detection algorithms
  - `analysis_results/`: Comparative analysis and visualizations
- `scripts/`: Execution scripts
- `src/`: Core project modules
  - `data/`: Data retrieval and preprocessing
  - `models/`: Anomaly detection algorithms
    - `isolation_forest.py`: Isolation Forest implementation
    - `lof.py`: Local Outlier Factor implementation
    - `cpp/`: C++ interface for AIDA
  - `utils/`: Utility functions
- `AIDA/`: Original AIDA algorithm implementation (C++)

## Troubleshooting

### AIDA Compilation Issues
If you encounter issues compiling AIDA:

1. **macOS Users**: Make sure OpenMP is properly installed:
   ```bash
   brew install libomp
   ```

2. **Windows Users**: Ensure MinGW or another C++ compiler with OpenMP support is in your PATH.

3. **If AIDA fails**: You can still run the Python-based algorithms:
   ```bash
   python scripts/run_all_algorithms.py --algorithms iforest lof temporal
   ```

### Data Issues
If you encounter issues with data loading or processing:

1. Check that the data files exist in the expected locations.
2. Ensure the CSV files have the expected format and column names.
3. Check the log output for specific error messages.

## Citing the Research
If you use this code in your research, please cite the bachelor's thesis:
Beerli, F. (2025). Unsupervised Anomaly Detection in S&P 500: A Comparative Approach. ZHAW – Zurich University of Applied Sciences.

## License
[Add your license information here]