# Unsupervised Anomaly Detection in S&P 500: A Comparative Approach

## 📈 Project Overview
This project performs a comparative analysis of three unsupervised anomaly detection algorithms — **AIDA**, **Isolation Forest**, and **Local Outlier Factor (LOF)** — applied to S&P 500 stock data. The project evaluates their effectiveness in identifying anomalous patterns in financial time series using three analysis approaches:

1. **Individual Subsequence Analysis**: Detecting anomalies in temporal windows of single stocks
2. **Cross-Index-Constituent Analysis**: Examining relationships between index anomalies and constituent stock behavior
3. **Multi-TS Matrix Analysis**: Analyzing anomalies across multiple stocks simultaneously (matrix-based approach)

---

## 📦 Prerequisites
- Python 3.8+
- C++ Compiler with OpenMP support (e.g., `g++`)
- Homebrew (for macOS users to install OpenMP)
- Recommended: Python virtual environment

---

## ⚙️ Setup and Installation

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. macOS Users: Install OpenMP Support (Required for AIDA)
```bash
brew install libomp
```

---

## 🚀 Workflow

### ✅ Step 1: Data Retrieval & Processing
Download and prepare S&P 500 index and constituent stock data.
```bash
# Retrieve raw data
python scripts/retrieve_data.py

# Process and create features, subsequences, and matrices
python scripts/prepare_data.py
```

### 🤖 Step 2: Run Complete Analysis
Execute the complete analysis workflow with one command:

```bash
# Run the complete analysis with default settings
python scripts/run_complete_analysis.py --run-all

# Customize with specific settings
python scripts/run_complete_analysis.py --run-individual-analysis --run-cross-analysis --run-multi-ts-analysis --window-sizes 3,5 --algorithms aida,iforest,lof
```

#### Alternative: Run Each Analysis Type Separately

```bash
# Individual Subsequence Analysis
python scripts/run_subsequence_algorithms.py --window-size 3 --overlap --algorithms all

# Cross-Index-Constituent Analysis
python scripts/run_constituent_relationship_analysis.py

# Multi-TS Matrix Analysis
python scripts/run_matrix_analysis.py
```

### 📊 Step 3: Compare and Visualize Results

```bash
# Compare subsequence anomalies
python scripts/compare_subsequence_anomalies.py --window-size 3 --overlap-type overlap

# Analyze constituent relationships
python scripts/analyze_constituent_relationships.py

# Compare multi-TS anomalies
python scripts/compare_multi_ts_anomalies.py --window-size 3 --overlap-type overlap --data data/processed/index_GSPC_processed.csv
```

---

## 🧠 Run Scripts with Convenience Helpers

The project includes convenience scripts for Windows and Unix systems:

### Windows (Batch Files)
```
scripts/run_analysis.bat          # Run complete analysis
scripts/run_constituent_analysis.bat  # Run constituent analysis
```

### Unix (Shell Scripts)
```
scripts/run_analysis.sh           # Run complete analysis
scripts/run_constituent_analysis.sh   # Run constituent analysis
```

---

## 📁 Directory Structure
```
data/
├── raw/                         # Original downloaded data
├── processed/                   # Preprocessed & feature-engineered data
│   ├── subsequences/            # Subsequence datasets
│   └── multi_ts/                # Multi-TS matrices
├── subsequence_results/         # Individual subsequence results
│   ├── aida/
│   ├── iforest/
│   └── lof/
├── constituent_analysis/        # Cross-index-constituent analysis
└── multi_ts_results/            # Matrix-based analysis results

src/
├── data/                        # Data retrieval and processing
├── models/                      # Algorithm implementations
│   ├── aida_helper.py           # AIDA integration
│   ├── isolation_forest.py      # Isolation Forest
│   ├── lof.py                   # Local Outlier Factor
│   └── cpp/                     # C++ implementations for AIDA
└── utils/                       # Utility functions

scripts/
├── run_complete_analysis.py     # Main entry point for all analyses
├── run_subsequence_algorithms.py # Individual subsequence analysis
├── run_constituent_relationship_analysis.py # Cross-analysis 
├── run_matrix_analysis.py       # Multi-TS matrix analysis
└── compare_*.py                 # Results visualization scripts
```

---

## 🛠️ Algorithms

### 🔹 AIDA (Analytic Isolation and Distance-based Anomaly)
A parameter-free anomaly detection method that combines distance and isolation principles, providing robust detection across different data distributions.

### 🔹 Isolation Forest
An efficient algorithm based on the principle that anomalies are easier to isolate in feature space, performing well with high-dimensional data.

### 🔹 Local Outlier Factor (LOF)
A density-based algorithm identifying local deviations in density, effective at detecting context-dependent anomalies.

---

## 🔎 Analysis Methods

### Individual Subsequence Analysis
Detects anomalies in temporal windows (subsequences) of individual stock time series, identifying unusual price and volume patterns.

### Cross-Index-Constituent Analysis
Examines how anomalies detected in the S&P 500 index relate to anomalies in constituent stocks, identifying market-wide vs. stock-specific patterns.

### Multi-TS Matrix Analysis
Treats multiple stock time series simultaneously as a matrix, detecting anomalies in collective market behavior that might be missed when looking at stocks individually.

---

## 🧾 Citation
> Beerli, F. (2025). *Unsupervised Anomaly Detection in S&P 500: A Comparative Approach*.  
> ZHAW – Zurich University of Applied Sciences.

---

## 🛠️ Troubleshooting

### AIDA Compilation Issues
- **macOS**: Make sure OpenMP is installed (`brew install libomp`)
- **Windows**: Use MinGW or another compiler with OpenMP support
- If issues persist, run with other algorithms: `--algorithms iforest lof`

### Common Issues
- Ensure data directories exist with proper permissions
- For OpenMP-related errors, verify that C++ compiler supports OpenMP
- If you see "Segmentation fault" with AIDA, try reducing the window size