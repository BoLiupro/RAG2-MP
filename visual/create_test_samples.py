"""
Create a random test sample CSV for visualization from train.csv.
Randomly samples valid trajectory windows from the training data.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import random


def create_test_samples(
    train_csv_path: str,
    output_csv_path: str,
    max_sample_num: int = 1000,
    obs_len: int = 12,
    pred_len: int = 1,
    random_seed: int = 42
):
    """
    Create a test sample CSV by randomly sampling from train.csv.
    
    Args:
        train_csv_path: Path to train.csv file
        output_csv_path: Path to output test CSV file
        max_sample_num: Maximum number of samples to extract
        obs_len: Observation trajectory length
        pred_len: Prediction trajectory length
        random_seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print("=" * 80)
    print("Creating Test Samples for Visualization")
    print("=" * 80)
    print(f"Train CSV: {train_csv_path}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Max samples: {max_sample_num}")
    print(f"Observation length: {obs_len}")
    print(f"Prediction length: {pred_len}")
    print(f"Random seed: {random_seed}")
    print("=" * 80)
    
    # Load train data
    print(f"\n[1/4] Loading train data...")
    df = pd.read_csv(train_csv_path)
    print(f"Loaded {len(df)} records from train.csv")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract valid samples
    print(f"\n[2/4] Extracting valid trajectory windows...")
    window_size = obs_len + pred_len
    valid_samples = []
    
    # Group by user_id
    for user_id, group in df.groupby('user_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        if len(group) < window_size:
            continue
        
        # Find all valid windows for this user
        for i in range(len(group) - window_size + 1):
            window = group.iloc[i:i + window_size]
            valid_samples.append({
                'user_id': user_id,
                'start_idx': i,
                'window_data': window
            })
    
    print(f"Found {len(valid_samples)} valid trajectory windows")
    
    # Random sampling
    print(f"\n[3/4] Randomly sampling {max_sample_num} windows...")
    if len(valid_samples) < max_sample_num:
        print(f"Warning: Only {len(valid_samples)} valid samples available, using all of them.")
        sampled_windows = valid_samples
    else:
        sampled_indices = random.sample(range(len(valid_samples)), max_sample_num)
        sampled_windows = [valid_samples[i] for i in sampled_indices]
    
    print(f"Sampled {len(sampled_windows)} windows")
    
    # Create output dataframe
    print(f"\n[4/4] Creating output CSV...")
    output_records = []
    
    for sample in sampled_windows:
        window_data = sample['window_data']
        
        # Add all records from this window
        for _, row in window_data.iterrows():
            output_records.append({
                'user_id': row['user_id'],
                'timestamp': row['timestamp'],
                'location_id': row['location_id']
            })
    
    # Create output dataframe
    output_df = pd.DataFrame(output_records)
    
    # Remove duplicates (in case of overlapping windows)
    output_df = output_df.drop_duplicates()
    
    # Sort by user_id and timestamp
    output_df = output_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Save to CSV
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"Saved {len(output_df)} records to {output_csv_path}")
    
    # Statistics
    unique_users = output_df['user_id'].nunique()
    unique_locations = output_df['location_id'].nunique()
    
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total records: {len(output_df)}")
    print(f"Unique users: {unique_users}")
    print(f"Unique locations: {unique_locations}")
    print(f"Sampled windows: {len(sampled_windows)}")
    print(f"Average records per user: {len(output_df) / unique_users:.1f}")
    print("=" * 80)
    
    # Verify samples can be used for prediction
    print(f"\n[Verification] Checking if samples can be used for prediction...")
    valid_prediction_samples = 0
    
    for user_id, group in output_df.groupby('user_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        valid_prediction_samples += max(0, len(group) - window_size + 1)
    
    print(f"✓ Can generate {valid_prediction_samples} prediction samples from the output CSV")
    print(f"  (obs_len={obs_len} + pred_len={pred_len} = window_size={window_size})")
    print("\n" + "=" * 80)
    print("Test samples created successfully!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Create random test samples from train.csv for visualization'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default='/workspace/China_Journal/data/beijing/train.csv',
        help='Path to train.csv file'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='/workspace/China_Journal/data/beijing/test_for_visual_beijing.csv',
        help='Path to output test CSV file'
    )
    parser.add_argument(
        '--max_sample_num',
        type=int,
        default=100,
        help='Maximum number of trajectory windows to sample'
    )
    parser.add_argument(
        '--obs_len',
        type=int,
        default=12,
        help='Observation trajectory length'
    )
    parser.add_argument(
        '--pred_len',
        type=int,
        default=1,
        help='Prediction trajectory length'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--city',
        type=str,
        default='beijing',
        choices=['beijing', 'shenzhen', 'nanchang'],
        help='City name (used for default paths)'
    )
    
    args = parser.parse_args()
    
    # Update default paths based on city if not explicitly specified
    if args.train_csv == '/workspace/China_Journal/data/beijing/train.csv' and args.city != 'beijing':
        args.train_csv = f'/workspace/China_Journal/data/{args.city}/train.csv'
    
    if args.output_csv == '/workspace/China_Journal/data/beijing/test_for_visual_beijing.csv':
        args.output_csv = f'/workspace/China_Journal/visual/test_for_visual_{args.city}.csv'
    
    # Create output directory if needed
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create test samples
    create_test_samples(
        train_csv_path=args.train_csv,
        output_csv_path=args.output_csv,
        max_sample_num=args.max_sample_num,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()
