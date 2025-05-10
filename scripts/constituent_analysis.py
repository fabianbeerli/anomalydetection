import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_combined_constituent_chart(output_dir, window_size, overlap_type, algorithms):
    # Load all summaries
    summaries = {}
    anomaly_indices = set()
    for algo in algorithms:
        summary_file = Path(output_dir) / f"cross_summary_{algo}_w{window_size}_{overlap_type}.json"
        if not summary_file.exists():
            continue
        with open(summary_file, "r") as f:
            summary = json.load(f)
            summaries[algo] = {s["sp500_anomaly_index"]: s["count"] for s in summary}
            anomaly_indices.update(summaries[algo].keys())
    if not summaries:
        print(f"No summaries found for window {window_size}, {overlap_type}")
        return
    anomaly_indices = sorted(anomaly_indices)
    x = np.arange(len(anomaly_indices))
    width = 0.25  # width of each bar

    # Define color mapping
    algo_colors = {
        "aida": "tab:red",
        "iforest": "tab:blue",
        "lof": "tab:green"
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, algo in enumerate(algorithms):
        counts = [summaries.get(algo, {}).get(idx, 0) for idx in anomaly_indices]
        color = algo_colors.get(algo, None)
        ax.bar(x + i*width, counts, width, label=algo, color=color)
    ax.set_xlabel("S&P500 Anomaly Index")
    ax.set_ylabel("Constituents with Anomaly")
    ax.set_title(f"Constituent Anomalies per S&P500 Anomaly (w{window_size}, {overlap_type})")
    ax.set_xticks(x + width * (len(algorithms)-1)/2)
    ax.set_xticklabels(anomaly_indices, rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"combined_constituent_bar_w{window_size}_{overlap_type}.png")
    plt.close()

def get_internal_ticker_name(ticker):
    return "sp500" if ticker.upper() == "GSPC" else ticker

def analyze_anomalies(sp500_anoms, constituent_dir, constituent_tickers, algo, window_size, overlap_type, results_dir, output_dir, plusminus1=False):
    summary = []
    for _, sp500_row in sp500_anoms.iterrows():
        idx = int(sp500_row['index'])
        anom_constituents = []
        for ticker in constituent_tickers:
            c_anom_file = results_dir / ticker / algo / f"w{window_size}_{overlap_type}" / f"{algo}_anomalies.csv"
            if not c_anom_file.exists():
                continue
            c_anoms = pd.read_csv(c_anom_file)
            if plusminus1:
                # ±1 subsequence window
                if ((c_anoms['index'] >= idx - 1) & (c_anoms['index'] <= idx + 1)).any():
                    anom_constituents.append(ticker)
            else:
                # Exact match
                if (c_anoms['index'] == idx).any():
                    anom_constituents.append(ticker)
        summary.append({
            "sp500_anomaly_index": idx,
            "sp500_anomaly_score": sp500_row['score'],
            "constituents_with_anomaly": anom_constituents,
            "count": len(anom_constituents)
        })
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsequence-results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--window-sizes", type=str, default="3")
    parser.add_argument("--only-overlap", action="store_true")
    parser.add_argument("--only-nonoverlap", action="store_true")
    parser.add_argument("--algorithms", type=str, default="aida,iforest,lof")
    args = parser.parse_args()

    window_sizes = [int(w) for w in args.window_sizes.split(",")]
    overlap_types = []
    if args.only_overlap:
        overlap_types = ["overlap"]
    elif args.only_nonoverlap:
        overlap_types = ["nonoverlap"]
    else:
        overlap_types = ["overlap", "nonoverlap"]
    algorithms = args.algorithms.split(",")

    results_dir = Path(args.subsequence_results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sp500_ticker = "GSPC"
    sp500_folder = get_internal_ticker_name(sp500_ticker)

    for window_size in window_sizes:
        for overlap_type in overlap_types:
            for algo in algorithms:
                sp500_anom_file = results_dir / sp500_folder / algo / f"w{window_size}_{overlap_type}" / f"{algo}_anomalies.csv"
                if not sp500_anom_file.exists():
                    print(f"Missing S&P500 anomaly file: {sp500_anom_file}")
                    continue
                sp500_anoms = pd.read_csv(sp500_anom_file)
                constituent_dir = results_dir
                constituent_tickers = [p.name for p in constituent_dir.iterdir() if p.is_dir() and p.name != sp500_folder]

                # --- Exact match (as before) ---
                summary = analyze_anomalies(
                    sp500_anoms, constituent_dir, constituent_tickers, algo, window_size, overlap_type, results_dir, output_dir, plusminus1=False
                )
                out_json = output_dir / f"cross_summary_{algo}_w{window_size}_{overlap_type}.json"
                with open(out_json, "w") as f:
                    json.dump(summary, f, indent=2)
                plt.figure(figsize=(10,4))
                plt.bar([s["sp500_anomaly_index"] for s in summary], [s["count"] for s in summary])
                plt.xlabel("S&P500 Anomaly Index")
                plt.ylabel("Constituents with Anomaly")
                plt.title(f"{algo} w{window_size}_{overlap_type}: #Constituents with Anomaly per S&P500 Anomaly")
                plt.tight_layout()
                plt.savefig(output_dir / f"cross_anomaly_bar_{algo}_w{window_size}_{overlap_type}.png")
                plt.close()

                
                # --- ±1 subsequence match ---
                pm1_dir = output_dir / "plusminus1"
                pm1_dir.mkdir(parents=True, exist_ok=True)
                summary_pm1 = analyze_anomalies(
                    sp500_anoms, constituent_dir, constituent_tickers, algo, window_size, overlap_type, results_dir, pm1_dir, plusminus1=True
                )
                out_json_pm1 = pm1_dir / f"cross_summary_{algo}_w{window_size}_{overlap_type}.json"
                with open(out_json_pm1, "w") as f:
                    json.dump(summary_pm1, f, indent=2)
                plt.figure(figsize=(10,4))
                plt.bar([s["sp500_anomaly_index"] for s in summary_pm1], [s["count"] for s in summary_pm1])
                plt.xlabel("S&P500 Anomaly Index")
                plt.ylabel("Constituents with Anomaly (±1)")
                plt.title(f"{algo} w{window_size}_{overlap_type}: #Constituents with Anomaly (±1) per S&P500 Anomaly")
                plt.tight_layout()
                plt.savefig(pm1_dir / f"cross_anomaly_bar_{algo}_w{window_size}_{overlap_type}.png")
                plt.close()

                # After all per-algo plots for this window_size and overlap_type:
                plot_combined_constituent_chart(output_dir, window_size, overlap_type, algorithms)
                pm1_dir = output_dir / "plusminus1"
                plot_combined_constituent_chart(pm1_dir, window_size, overlap_type, algorithms)


if __name__ == "__main__":
    main()