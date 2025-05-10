import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

# Settings
algos = ['aida', 'iforest', 'lof']
shapes = ['o', 's', '^']
colors = ['tab:red', 'tab:blue', 'tab:green']
offsets = [40, -80, 120]  # y-offsets for each algo
base_dir = Path("data/analysis_results/true_or_false_positives")
sp500_file = Path("data/processed/index_GSPC_processed.csv")




def load_anomalies_and_truth(algo, overlap_type, multi_ts=False):
    folder = base_dir / algo / overlap_type
    if multi_ts:
        anomalies_file = folder / f"{algo}_multi_ts_anomalies.csv"
        truth_file = folder / "trueorfalse_positives_multi_ts.json"
    else:
        anomalies_file = folder / f"{algo}_anomalies.csv"
        truth_file = folder / "trueorfalse_positives.json"
    if not anomalies_file.exists() or not truth_file.exists():
        print(f"Missing files for {algo} {overlap_type} (multi_ts={multi_ts}):")
        print(f"  {anomalies_file.exists()=}, {anomalies_file}")
        print(f"  {truth_file.exists()=}, {truth_file}")
        return None, None
    anomalies = pd.read_csv(anomalies_file)
    print(f"Loading JSON: {truth_file}")  # Log the file path
    with open(truth_file, "r") as f:
        truth = json.load(f)
    return anomalies, truth

def plot_for_overlap_type(overlap_type, save_path, multi_ts=False):
    sp500 = pd.read_csv(sp500_file, parse_dates=["Date"])
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(sp500["Date"], sp500["Close"], label="S&P500", color="black", linewidth=1)

    for i, algo in enumerate(algos):
        anomalies, truth = load_anomalies_and_truth(algo, overlap_type, multi_ts=multi_ts)
        if anomalies is None or truth is None:
            continue
        y_offset = offsets[i]
        tp_count = 0
        fp_count = 0
        for idx, row in anomalies.iterrows():
            # Use correct column for index
            subseq_idx = row['window_idx'] if multi_ts else row['subsequence_idx']
            date = pd.to_datetime(row["start_date"])
            y_base = sp500.loc[sp500["Date"] == date, "Close"]
            if y_base.empty:
                continue
            y_base = y_base.values[0]
            y = y_base + y_offset
            # Draw dotted line from offset marker to original close value
            ax.plot([date, date], [y_base, y], color=colors[i], linestyle='dotted', linewidth=1, zorder=2)
            anomaly_key = f"anomaly_{int(subseq_idx)}"
            if anomaly_key not in truth:
                print(f"Warning: No entry for {anomaly_key} in {algo} {overlap_type} true/false positives file (multi_ts={multi_ts}).")
                is_true = None
            else:
                is_true = truth[anomaly_key].get("is_true_positive", None)
            marker = shapes[i]
            color = colors[i]
            # Only add label for the first anomaly (for legend)
            label = None
            if idx == 0:
                label = f"{algo} (TP: {{TP}}, FP: {{FP}})"  # Placeholder, will replace later
            if is_true is True:
                tp_count += 1
                ax.scatter(date, y, marker=marker, color=color, s=80, label=label, zorder=3, alpha=1.0, edgecolor='black', linewidth=1)
            elif is_true is False:
                fp_count += 1
                ax.scatter(date, y, marker=marker, color=color, s=80, label=label, zorder=3, alpha=0.25, edgecolor='gray', linewidth=1)
            else:
                ax.scatter(date, y, marker=marker, color=color, s=80, label=label, zorder=3, alpha=0.6, edgecolor='black', linewidth=1)
        # Update legend label with counts
        for lh in ax.collections:
            if lh.get_label() and lh.get_label().startswith(algo):
                lh.set_label(f"{algo} (TP: {tp_count}, FP: {fp_count})")
    ax.set_title(f"S&P500 Anomalies ({overlap_type}{' multi_ts' if multi_ts else ''})")
    ax.set_ylabel("Close (offset for visibility)")
    ax.set_xlabel("Date")
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def main():
    plot_for_overlap_type("overlap", "sp500_anomalies_overlap.png")
    plot_for_overlap_type("nonoverlap", "sp500_anomalies_nonoverlap.png")
    plot_for_overlap_type("overlap", "sp500_anomalies_overlap_multi_ts.png", multi_ts=True)
    plot_for_overlap_type("nonoverlap", "sp500_anomalies_nonoverlap_multi_ts.png", multi_ts=True)
    print("Plots saved as sp500_anomalies_overlap.png, sp500_anomalies_nonoverlap.png, sp500_anomalies_overlap_multi_ts.png, and sp500_anomalies_nonoverlap_multi_ts.png")


if __name__ == "__main__":
    main()