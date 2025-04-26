import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

def get_internal_ticker_name(ticker):
    # Map GSPC to sp500 for file/folder naming
    return "sp500" if ticker.upper() == "GSPC" else ticker

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

    # Use mapping for S&P500 folder
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
                summary = []
                for _, sp500_row in sp500_anoms.iterrows():
                    idx = int(sp500_row['index'])
                    anom_constituents = []
                    for ticker in constituent_tickers:
                        c_anom_file = results_dir / ticker / algo / f"w{window_size}_{overlap_type}" / f"{algo}_anomalies.csv"
                        if not c_anom_file.exists():
                            continue
                        c_anoms = pd.read_csv(c_anom_file)
                        if (c_anoms['index'] == idx).any():
                            anom_constituents.append(ticker)
                    summary.append({
                        "sp500_anomaly_index": idx,
                        "sp500_anomaly_score": sp500_row['score'],
                        "constituents_with_anomaly": anom_constituents,
                        "count": len(anom_constituents)
                    })
                # Save summary
                out_json = output_dir / f"cross_summary_{algo}_w{window_size}_{overlap_type}.json"
                with open(out_json, "w") as f:
                    json.dump(summary, f, indent=2)
                # Visualization: bar plot of anomaly counts
                plt.figure(figsize=(10,4))
                plt.bar([s["sp500_anomaly_index"] for s in summary], [s["count"] for s in summary])
                plt.xlabel("S&P500 Anomaly Index")
                plt.ylabel("Constituents with Anomaly")
                plt.title(f"{algo} w{window_size}_{overlap_type}: #Constituents with Anomaly per S&P500 Anomaly")
                plt.tight_layout()
                plt.savefig(output_dir / f"cross_anomaly_bar_{algo}_w{window_size}_{overlap_type}.png")
                plt.close()

if __name__ == "__main__":
    main()