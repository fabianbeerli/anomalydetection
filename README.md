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

## ✅ True/False Positive Visualization

To run the true/false positive visualization, follow these steps:

1. **Create the Folder Structure**

   In your `data` directory, manually create the following folder structure:

   ```
   data/
     true_or_false_positives/
       aida/
         overlap/
         nonoverlap/
       iforest/
         overlap/
         nonoverlap/
       lof/
         overlap/
         nonoverlap/
   ```

2. **Copy Anomaly CSV Results**

   For each algorithm (`aida`, `iforest`, `lof`), copy the relevant CSV anomaly results from both the `subsequence_results` and `multi_ts_results` folders into the corresponding `overlap` and `nonoverlap` subfolders you just created.

3. **Prepare the JSON Annotation File**

   In each `overlap` and `nonoverlap` subfolder, create a JSON file (e.g., `trueorfalse_positives_multi_ts.json`) containing all anomalies and their true/false positive annotations.  
   You can use the currently existing file in that location as a template for the format.

4. **Run the Visualization Script**

   Once the folders and files are prepared, run:
   ```bash
   python scripts/true_or_false_positives.py
   ```
   This will generate the true/false positive visualizations based on your annotations.

> **Note:** The script expects the folder and file structure as described above. Make sure your JSON files are formatted correctly and contain all required anomaly annotations.

---