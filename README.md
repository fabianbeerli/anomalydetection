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
#### A. AIDA Anomaly Detection
**Script:** `scripts/run_aida-detection.sh` (Unix) or `scripts/run_aida_detection.bat` (Windows)
**Purpose:** Run AIDA anomaly detection on S&P 500 data
**Output:** 
- Anomaly scores file
- Detected anomalies file
- Visualizations in `data/aida_results/`
```bash
# Unix
./scripts/run_aida-detection.sh

# Windows
scripts\run_aida_detection.bat
```

#### B. Alternative Detection Methods (Optional)
Additional anomaly detection methods can be implemented in separate scripts. The project structure supports:
- Isolation Forest
- Local Outlier Factor (LOF)
- Deep learning-based approaches

## Project Structure
- `data/`
  - `raw/`: Original downloaded data
  - `processed/`: Preprocessed and feature-engineered data
- `scripts/`: Execution scripts
- `src/`: Core project modules
  - `data/`: Data retrieval and preprocessing
  - `models/`: Anomaly detection algorithms
  - `utils/`: Utility functions
- `AIDA/`: AIDA algorithm implementation

## Citing the Research
If you use this code in your research, please cite the bachelor's thesis:
Beerli, F. (2025). Unsupervised Anomaly Detection in S&P 500: A Comparative Approach. ZHAW – Zurich University of Applied Sciences.

## License
[Add your license information here]