"""
Dataset Module for Mobility Prediction
Loads and preprocesses trajectory data for training, validation, and testing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
import os


class MobilityDataset(Dataset):
    """
    PyTorch Dataset for mobility trajectory data.
    Loads processed CSV files and creates trajectory samples.
    """
    
    def __init__(
        self,
        data_path: str,
        city: str,
        obs_len: int = 12,
        pred_len: int = 1,
        poi_data_path: str = None
    ):
        """
        Initialize the mobility dataset.
        
        Args:
            data_path: Path to CSV file (train.csv, val.csv, or test.csv)
            city: City name ('beijing', 'shenzhen', 'nanchang')
            obs_len: Length of observation trajectory
            pred_len: Length of prediction trajectory
            poi_data_path: Path to POI data CSV file
        """
        self.data_path = data_path
        self.city = city
        self.obs_len = obs_len
        self.pred_len = pred_len
        
        # Load POI data if provided
        self.poi_data = None
        if poi_data_path and os.path.exists(poi_data_path):
            self.poi_data = self._load_poi_data(poi_data_path)
        
        # Load and process trajectory data
        self.samples = self._load_trajectory_data()
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_poi_data(self, poi_path: str) -> Dict[int, Dict[str, float]]:
        """
        Load POI data from CSV file.
        
        Args:
            poi_path: Path to POI CSV file
        
        Returns:
            Dictionary mapping location_id to POI features
        """
        print(f"Loading POI data from {poi_path}...")
        poi_df = pd.read_csv(poi_path)
        
        poi_data = {}
        poi_columns = [col for col in poi_df.columns if col.endswith('_percentage')]
        
        for _, row in poi_df.iterrows():
            grid_id = row['grid_id']
            poi_features = {}
            for col in poi_columns:
                poi_type = col.replace('_percentage', '')
                poi_features[poi_type] = row[col]
            poi_data[grid_id] = poi_features
        
        print(f"Loaded POI data for {len(poi_data)} locations")
        return poi_data
    
    def _load_trajectory_data(self) -> List[Dict[str, Any]]:
        """
        Load trajectory data and create samples with sliding window.
        
        Returns:
            List of trajectory samples
        """
        print(f"Loading trajectory data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        samples = []
        window_size = self.obs_len + self.pred_len
        
        # Group by user_id
        for user_id, group in df.groupby('user_id'):
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            if len(group) < window_size:
                continue
            
            # Sliding window
            for i in range(len(group) - window_size + 1):
                window = group.iloc[i:i + window_size]
                
                # Create observation trajectory
                obs_trajectory = []
                for j in range(self.obs_len):
                    obs_trajectory.append({
                        'location_id': int(window.iloc[j]['location_id']),
                        'timestamp': str(window.iloc[j]['timestamp'])
                    })
                
                # Create prediction target(s)
                pred_trajectory = []
                for j in range(self.obs_len, window_size):
                    pred_trajectory.append({
                        'location_id': int(window.iloc[j]['location_id']),
                        'timestamp': str(window.iloc[j]['timestamp'])
                    })
                
                sample = {
                    'user_id': user_id,
                    'observation': obs_trajectory,
                    'prediction': pred_trajectory,
                    'next_location': int(window.iloc[self.obs_len]['location_id'])
                }
                
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
        
        Returns:
            Sample dictionary with observation, prediction, and metadata
        """
        return self.samples[idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        # Count unique users
        unique_users = set(sample['user_id'] for sample in self.samples)
        
        # Count unique locations
        unique_locations = set()
        for sample in self.samples:
            for point in sample['observation']:
                unique_locations.add(point['location_id'])
            unique_locations.add(sample['next_location'])
        
        # Get location distribution
        location_counts = {}
        for sample in self.samples:
            loc = sample['next_location']
            location_counts[loc] = location_counts.get(loc, 0) + 1
        
        # Get temporal distribution (hour of day)
        hour_counts = {}
        for sample in self.samples:
            timestamp = sample['observation'][-1]['timestamp']
            try:
                hour = int(timestamp.split(' ')[1].split(':')[0])
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            except:
                pass
        
        stats = {
            'total_samples': len(self.samples),
            'unique_users': len(unique_users),
            'unique_locations': len(unique_locations),
            'obs_len': self.obs_len,
            'pred_len': self.pred_len,
            'city': self.city,
            'top_locations': sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'hour_distribution': dict(sorted(hour_counts.items()))
        }
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()
        
        print(f"\n{'='*70}")
        print(f"Dataset Statistics - {self.city.upper()}")
        print(f"{'='*70}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Unique users: {stats['unique_users']}")
        print(f"Unique locations: {stats['unique_locations']}")
        print(f"Observation length: {stats['obs_len']}")
        print(f"Prediction length: {stats['pred_len']}")
        print(f"\nTop 10 most frequent next locations:")
        for loc, count in stats['top_locations']:
            print(f"  Location {loc}: {count} occurrences")
        print(f"\nHourly distribution (top 5):")
        sorted_hours = sorted(stats['hour_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
        for hour, count in sorted_hours:
            print(f"  Hour {hour:02d}:00: {count} samples")
        print(f"{'='*70}\n")


def load_datasets(
    city: str,
    data_dir: str = "/workspace/China_Journal/data",
    obs_len: int = 12,
    pred_len: int = 1
) -> Tuple[MobilityDataset, MobilityDataset, MobilityDataset]:
    """
    Load train, validation, and test datasets for a city.
    
    Args:
        city: City name ('beijing', 'shenzhen', 'nanchang')
        data_dir: Base directory containing city data
        obs_len: Observation length
        pred_len: Prediction length
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    city_dir = os.path.join(data_dir, city)
    poi_path = os.path.join(city_dir, "poi.csv")
    
    # Check if POI file exists
    if not os.path.exists(poi_path):
        print(f"Warning: POI file not found at {poi_path}")
        poi_path = None
    
    # Load datasets
    train_dataset = MobilityDataset(
        data_path=os.path.join(city_dir, "train.csv"),
        city=city,
        obs_len=obs_len,
        pred_len=pred_len,
        poi_data_path=poi_path
    )
    
    val_dataset = MobilityDataset(
        data_path=os.path.join(city_dir, "val.csv"),
        city=city,
        obs_len=obs_len,
        pred_len=pred_len,
        poi_data_path=poi_path
    )
    
    test_dataset = MobilityDataset(
        data_path=os.path.join(city_dir, "small_test.csv"),
        city=city,
        obs_len=obs_len,
        pred_len=pred_len,
        poi_data_path=poi_path
    )
    
    # Print statistics
    print("\n" + "="*70)
    print("DATASET LOADING SUMMARY")
    print("="*70)
    train_dataset.print_statistics()
    val_dataset.print_statistics()
    test_dataset.print_statistics()
    
    return train_dataset, val_dataset, test_dataset
