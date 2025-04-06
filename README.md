# Unsupervised Anomaly Detection in S&P 500: A Comparative Approach

This project implements unsupervised anomaly detection methods (AIDA, Isolation Forest, and LOF) to identify abnormal patterns in S&P 500 index and constituent stocks data.

## Overview

The goal of this project is to compare the performance of three unsupervised anomaly detection algorithms:

1. **AIDA** (Analytic Isolation and Distance-based Anomaly)
2. **Isolation Forest**
3. **Local Outlier Factor (LOF)**

The project analyzes daily data from the S&P 500 index and selected constituent stocks to identify market anomalies, validate them against financial news events, and evaluate each algorithm's explanatory capabilities.

## Project Structure

```
anomaly_detection_sp500/
│
├── data/
│   ├── raw/              # Raw data downloaded from Yahoo Finance
│   └── processed/        # Processed and feature-engineered data
│
├── src/                  # Source code
│   ├── config.py         # Configuration parameters
│   ├── data/             # Data retrieval and preparation
│   ├── models/           # Anomaly detection models
│   ├── visualization/    # Visualization utilities
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Utility functions
│
├── notebooks/            # Jupyter notebooks
├── scripts/              # Executable scripts
├── tests/                # Unit tests
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone 
cd anomaly_detection_sp500
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Data Retrieval and Preparation

1. Retrieve S&P 500 index data and constituent stocks:
```bash
python scripts/retrieve_data.py
```

2. Process and prepare the data for anomaly detection:
```bash
python scripts/prepare_data.py
```

## Usage

### Module-Based Execution

Each part of the project can be run independently:

```python
# Data retrieval
from src.data.retrieval import retrieve_sp500_index_data, retrieve_constituent_data
retrieve_sp500_index_data()
retrieve_constituent_data()

# Data preparation
from src.data.preparation import process_all_data
process_all_data()

# (Future) Anomaly detection
# from src.models.aida import detect_anomalies_aida
# detect_anomalies_aida(data)
```

### Script-Based Execution

The project includes several scripts for common operations:

```bash
# Data retrieval
python scripts/retrieve_data.py

# Data preparation
python scripts/prepare_data.py

# (Future) Anomaly detection
# python scripts/detect_anomalies.py
```

## Features

- Data retrieval from Yahoo Finance
- Comprehensive data preprocessing and feature engineering
- Time series subsequence creation for anomaly detection
- Multi-stock subsequence matrices for cross-asset analysis
- (Future) Implementation of AIDA, Isolation Forest, and LOF algorithms
- (Future) Visualization and comparison of detected anomalies
- (Future) Validation against financial news events

## License

This project is part of a bachelor's thesis at ZHAW – Zurich University of Applied Sciences.

## Acknowledgments

- Cirillo Pasquale (Thesis Supervisor)
- ZHAW School of Management and Law