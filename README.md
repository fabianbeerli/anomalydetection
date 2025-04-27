# Unsupervised Anomaly Detection in S&P 500: A Comparative Approach

## 📈 Project Overview
This project performs a comparative analysis of three unsupervised anomaly detection algorithms — **AIDA**, **Isolation Forest**, and **Local Outlier Factor (LOF)** — applied to S&P 500 stock data. The project evaluates their effectiveness in identifying anomalous patterns in financial time series using several analysis approaches.

---

## ⚙️ Quick Start

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (macOS only) Install OpenMP Support (Required for AIDA)
```bash
brew install libomp
```

---

## 🚀 Run the Analysis

### Step 1: Retrieve Data
```bash
python scripts/retrieve_data.py
```

### Step 2: Prepare Data
```bash
python scripts/prepare_data.py
```

### Step 3: Run Complete Analysis
```bash
python scripts/run_complete_analysis.py --run-all
```

> There are additional options and analysis modes available; see the code for more details.

---