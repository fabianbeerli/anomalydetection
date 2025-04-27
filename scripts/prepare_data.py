#!/usr/bin/env python
"""
Script to process and prepare data for anomaly detection.
"""
import os
import sys
import logging
import json
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preparation import (
    process_all_data,
    load_ticker_data,
    create_subsequence_dataset,
    create_multi_ts_subsequences,
    save_subsequence_dataset
)
from src import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to process and prepare data.
    """
    logger.info("Starting data preparation process")
    
    # Process all data
    processed_data = process_all_data()
    
    if processed_data.get('sp500_processed'):
        logger.info(f"S&P 500 index data processed and saved to {processed_data['sp500_processed']}")
    else:
        logger.warning("No processed S&P 500 index data available")
    
    if processed_data.get('constituent_processed'):
        logger.info(f"Processed {len(processed_data['constituent_processed'])} constituent stock files")
    else:
        logger.warning("No processed constituent stock data available")
    
    # Create subsequence directory
    subseq_dir = config.PROCESSED_DATA_DIR / 'subsequences'
    subseq_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subsequences for the S&P 500 index
    if processed_data.get('sp500_processed'):
        sp500_df = load_ticker_data(processed_data['sp500_processed'])
        
        if sp500_df is not None:

                sp500_features = sp500_df
                
                for length in [3, 5, 10]:
                    logger.info(f"Creating subsequences of length {length} for S&P 500 index")
                    
                    subseqs_overlap = create_subsequence_dataset(
                        sp500_features, 
                        subsequence_length=length, 
                        step=1
                    )
                    
                    save_subsequence_dataset(
                        subseqs_overlap,
                        subseq_dir,
                        prefix=f'sp500_len{length}_overlap'
                    )
                    
                    subseqs_nonoverlap = create_subsequence_dataset(
                        sp500_features, 
                        subsequence_length=length, 
                        step=length
                    )
                    
                    save_subsequence_dataset(
                        subseqs_nonoverlap,
                        subseq_dir,
                        prefix=f'sp500_len{length}_nonoverlap'
                    )
    
    # Create individual subsequences for each constituent stock
    if processed_data.get('constituent_processed'):
        for file_path in processed_data['constituent_processed']:
            ticker = Path(file_path).stem.replace('_processed', '')
            df = load_ticker_data(file_path)
            
            if df is not None and not df.empty:
                
                    df_features = df

                    for length in [3]:
                        logger.info(f"Creating subsequences of length {length} for ticker {ticker}")

                        subseqs_overlap = create_subsequence_dataset(
                            df_features,
                            subsequence_length=length,
                            step=1
                        )
                        
                        save_subsequence_dataset(
                            subseqs_overlap,
                            subseq_dir,
                            prefix=f'{ticker}_len{length}_overlap'
                        )
                        
                        subseqs_nonoverlap = create_subsequence_dataset(
                            df_features,
                            subsequence_length=length,
                            step=length
                        )

                        save_subsequence_dataset(
                            subseqs_nonoverlap,
                            subseq_dir,
                            prefix=f'{ticker}_len{length}_nonoverlap'
                        )
                
            else:
                logger.warning(f"No data available for {ticker} or DataFrame is empty.")

    # Create multi-TS subsequences for analysis across multiple stocks
    if processed_data.get('constituent_processed') and len(processed_data['constituent_processed']) > 0:
        constituent_dfs = []
        constituent_tickers = []

        for file_path in processed_data['constituent_processed'][:10]:
            ticker = Path(file_path).stem.replace('_processed', '')
            df = load_ticker_data(file_path)

            if df is not None:
                if not df.empty:
                    constituent_dfs.append(df)
                    constituent_tickers.append(ticker)
                else:
                    logger.warning(f"Empty DataFrame for {ticker}, skipping")

        if constituent_dfs:
            feature_columns = [
                'daily_return',
                'volume_change',
                'high_low_range'
            ]

            valid_dfs = []
            valid_tickers = []

            for i, (df, ticker) in enumerate(zip(constituent_dfs, constituent_tickers)):
                available_features = [col for col in feature_columns if col in df.columns]

                if available_features:
                    valid_dfs.append(df[available_features])
                    valid_tickers.append(ticker)
                else:
                    logger.warning(f"No required features available for {ticker}. Available columns: {df.columns.tolist()}")

            if valid_dfs:
                logger.info(f"Creating multi-TS subsequences for {len(valid_dfs)} stocks")

                try:
                    multi_ts_dir = config.PROCESSED_DATA_DIR / 'multi_ts'
                    multi_ts_dir.mkdir(exist_ok=True, parents=True)

                    subsequence_configs = [
                        (3, 1, 'overlap'),
                        (3, 3, 'nonoverlap'),
                        (5, 1, 'overlap'),
                        (5, 5, 'nonoverlap')
                    ]

                    for length, step, overlap_label in subsequence_configs:
                        logger.info(f"Creating multi-TS subsequences with length={length}, step={step} ({overlap_label})")

                        multi_ts_subseqs = create_multi_ts_subsequences(
                            valid_dfs,
                            valid_tickers,
                            [col for col in feature_columns if col in valid_dfs[0].columns],
                            subsequence_length=length,
                            step=step
                        )

                        save_subsequence_dataset(
                            multi_ts_subseqs,
                            multi_ts_dir,
                            prefix=f'multi_ts_len{length}_{overlap_label}'
                        )

                        logger.info(f"Saved {len(multi_ts_subseqs)} multi-TS subsequences with length={length}, {overlap_label}")

                except Exception as e:
                    logger.error(f"Error creating multi-TS subsequences: {e}")
            else:
                logger.warning("No valid DataFrames with required features for multi-TS analysis")
    
    logger.info("Data preparation and subsequence creation completed")


if __name__ == "__main__":
    main()