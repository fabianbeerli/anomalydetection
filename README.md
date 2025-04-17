# Unsupervised Anomaly Detection in S&P 500: A Comparative Approach

## Project Overview
This project implements a comparative analysis of anomaly detection algorithms (AIDA, Isolation Forest, and Local Outlier Factor) on S&P 500 market data.

## Prerequisites
- Python 3.8+
- C++ Compiler (g++)
- OpenMP support
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

#### Option A: Run All Algorithms
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

#### Option B: Individual Algorithm Execution

##### AIDA Anomaly Detection
**Script:** `scripts/run_aida-detection.sh` (Unix) or `scripts/run_aida_detection.bat` (Windows)  
**Purpose:** Run AIDA anomaly detection on S&P 500 data  
**Output:** Anomaly scores and detected anomalies in `data/aida_results/`
```bash
# Unix
./scripts/run_aida-detection.sh

# Windows
scripts\run_aida_detection.bat
```

##### Isolation Forest & Local Outlier Factor
These algorithms are implemented in Python and can be run via the run_all_algorithms.py script with specific algorithm selection:
```bash
# Run only Isolation Forest
python scripts/run_all_algorithms.py --algorithms iforest

# Run only Local Outlier Factor
python scripts/run_all_algorithms.py --algorithms lof
```

### Step 4: Time Series Analysis
Temporal analysis is supported for both Isolation Forest and LOF algorithms. This allows for sliding window anomaly detection, which is particularly useful for identifying anomalies that occur over sequences of trading days.

```bash
# Run temporal analysis with default window size (5) and step (1)
python scripts/run_all_algorithms.py --algorithms temporal

# Run with custom window size and step
python scripts/run_all_algorithms.py --algorithms temporal --window-size 3 --step 1
```

The temporal analysis creates overlapping subsequences (e.g., days 1,2,3 compared to 2,3,4, etc.) which helps identify anomalies in the time series patterns rather than just individual data points.

## Project Structure
- `data/`
  - `raw/`: Original downloaded data
  - `processed/`: Preprocessed and feature-engineered data
  - `algorithm_results/`: Results from running anomaly detection algorithms
- `scripts/`: Execution scripts
- `src/`: Core project modules
  - `data/`: Data retrieval and preprocessing
  - `models/`: Anomaly detection algorithms
    - `isolation_forest.py`: Isolation Forest implementation
    - `lof.py`: Local Outlier Factor implementation
  - `utils/`: Utility functions
- `AIDA/`: AIDA algorithm implementation (C++)

## Citing the Research
If you use this code in your research, please cite the bachelor's thesis:
Beerli, F. (2025). Unsupervised Anomaly Detection in S&P 500: A Comparative Approach. ZHAW – Zurich University of Applied Sciences.

## License
[Add your license information here]