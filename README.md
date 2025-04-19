
# Unsupervised Anomaly Detection in S&P 500: A Comparative Approach

## 📈 Project Overview
This project performs a comparative analysis of three unsupervised anomaly detection algorithms — **AIDA**, **Isolation Forest**, and **Local Outlier Factor (LOF)** — applied to subsequences derived from S&P 500 stock data. The aim is to evaluate their effectiveness and performance in identifying anomalous patterns in financial time series.

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

## 🚀 Workflow Steps

### ✅ Step 1: Data Retrieval
Download S&P 500 index and constituent stock data.
```bash
python scripts/retrieve_data.py
```

### 🔧 Step 2: Data Preprocessing
Process raw data and engineer features. This generates subsequence datasets.
```bash
python scripts/prepare_data.py
```

### 🤖 Step 3: Run Anomaly Detection
Execute anomaly detection algorithms with custom window sizes and overlap settings.

```bash
# All algorithms, window size 3, with overlap
python scripts/run_subsequence_algorithms.py --window-size 3 --overlap --algorithms all

# All algorithms, window size 3, without overlap
python scripts/run_subsequence_algorithms.py --window-size 3 --no-overlap --algorithms all

# Specific algorithms (e.g., AIDA and LOF), window size 5
python scripts/run_subsequence_algorithms.py --window-size 5 --overlap --algorithms aida lof

# Larger window size
python scripts/run_subsequence_algorithms.py --window-size 10 --overlap --algorithms all
```

#### Parameters:
- `--window-size`: Size of the subsequence window (default: 3)
- `--overlap` / `--no-overlap`: Control overlapping behavior (default: overlap)
- `--algorithms`: Choose from `aida`, `iforest`, `lof`, or `all`
- `--subsequence-dir`: Optional path to precomputed subsequences
- `--output`: Optional directory to save results

---

### 📊 Step 4: Analyze and Compare Results
Visualize anomalies and compare detection performance.

```bash
# Overlapping windows, size 3
python scripts/compare_subsequence_anomalies.py --results-base data/subsequence_results --window-size 3 --overlap-type overlap

# Non-overlapping windows, size 3
python scripts/compare_subsequence_anomalies.py --results-base data/subsequence_results --window-size 3 --overlap-type nonoverlap

# Larger window, with output directory
python scripts/compare_subsequence_anomalies.py --results-base data/subsequence_results --window-size 10 --overlap-type overlap --output data/analysis_results/w10_overlap
```

#### Parameters:
- `--results-base`: Base directory for results (default: `data/subsequence_results`)
- `--window-size`: Window size used for comparison (default: 3)
- `--overlap-type`: `overlap` or `non-overlap`
- `--data`: Optional path to raw data
- `--output`: Optional directory to save analysis results

---

## 📁 Directory Structure
```
data/
├── raw/                         # Original downloaded data
├── processed/                   # Preprocessed & feature-engineered data
│   └── subsequences/            # Subsequence datasets
├── subsequence_results/         # Results from algorithms
│   ├── w3_overlap/
│   │   ├── aida/
│   │   ├── iforest/
│   │   └── lof/
│   ├── w3_nonoverlap/
│   └── ...
└── analysis_results/            # Comparative visualizations & summaries
```

---

## 📄 Output Files

- `*_scores.dat`: Raw anomaly scores
- `*_anomalies.csv`: Detected anomalies and subsequence info
- `*_execution_time.txt`: Timing for each algorithm
- `subsequence_anomalies_comparison.png`: Anomaly visualization
- `anomaly_detection_detailed_summary.txt`: Statistical summary

---

## 🧠 About the Algorithms

### 🔹 AIDA (Analytic Isolation and Distance-based Anomaly)
A parameter-free anomaly detection method combining distance and isolation principles. Robust and versatile across different data types.

### 🔹 Isolation Forest
Based on the principle that anomalies are easier to isolate. Efficient for high-dimensional data.

### 🔹 Local Outlier Factor (LOF)
Density-based algorithm that highlights deviations in local neighborhood density. Ideal for datasets with variable density.

---

## 🛠️ Troubleshooting

### AIDA Compilation Issues
- **macOS**: Make sure OpenMP is installed (`brew install libomp`)
- **Windows**: Use MinGW or another compiler with OpenMP support
- If issues persist, run with other algorithms:  
  ```bash
  --algorithms iforest lof
  ```

### Data Issues
- Ensure data files exist in expected paths
- Check for proper CSV formatting and required columns
- Review terminal/log output for error messages

---

## 📚 Citation
> Beerli, F. (2025). *Unsupervised Anomaly Detection in S&P 500: A Comparative Approach*.  
> ZHAW – Zurich University of Applied Sciences.

---

## 🧾 License
This project is provided for academic and research purposes. For licensing details, please consult the LICENSE file if present.
