"""
RAG Database Builder
Constructs the experience pool database for RAG-based mobility prediction.

Steps:
1. Read train.csv data
2. Generate embeddings for each trajectory sample using LLM
3. Normalize embeddings, apply PCA, and perform HDBSCAN clustering (target: ~1k clusters)
4. Select representative samples from each cluster (center, edge, extreme samples)
5. Apply functional coverage constraints (origin function, time, spatial, distance)
6. Save the final RAG database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import json
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
import faiss

from model.LLM import MobilityLLM
from util.utils import grid_id_to_latlon, haversine_distance


class RAGDatabaseBuilder:
    """
    Builder for constructing the RAG experience pool database.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        city: str,
        output_dir: str = "/workspace/China_Journal/util/rag_database",
        target_clusters: int = 1000,
        model_name: str = "",
        samples_per_cluster: int = 10
    ):
        """
        Initialize the RAG database builder.
        
        Args:
            config: Configuration dictionary
            city: City name ('beijing', 'shenzhen', 'nanchang')
            output_dir: Output directory for the RAG database
            target_clusters: Target number of clusters (~1k)
            samples_per_cluster: Number of representative samples per cluster
        """
        self.config = config
        self.city = city
        self.model_name = model_name
        self.output_dir = os.path.join(output_dir,self.model_name,city)
        self.target_clusters = target_clusters
        self.samples_per_cluster = samples_per_cluster
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize LLM for embedding generation
        print("\n" + "="*80)
        print("Initializing LLM for trajectory encoding...")
        print("="*80)
        
        llm_config = config['model']['llm']
        self.llm = MobilityLLM(
            model_name=llm_config['model_name'],
            model_path=llm_config['model_path'],
            use_quantization=llm_config.get('use_quantization', True),
            use_lora=False,  # No LoRA for encoding
        )
        
        print(f"LLM initialized: {llm_config['model_name']}")
        
        # Load POI data (always use default path)
        data_config = config['data']
        poi_path = f"/workspace/China_Journal/data/{city}/poi.csv"
        
        print(f"Loading POI data from: {poi_path}")
        self.poi_data = self._load_poi_data(poi_path)
        print(f"Loaded POI data for {len(self.poi_data)} grids")
        
    def _load_poi_data(self, poi_path: str) -> Dict[int, Dict[str, float]]:
        """Load POI data from CSV file."""
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
        
        return poi_data
    
    def load_training_data(self, train_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
        """
        Load training data from CSV file.
        
        Args:
            train_path: Path to train.csv
            max_samples: Maximum number of samples to load (None = all)
        
        Returns:
            List of trajectory samples
        """
        print("\n" + "="*80)
        print("Loading training data...")
        print("="*80)
        print(f"Reading from: {train_path}")
        
        df = pd.read_csv(train_path)
        print(f"Total records in CSV: {len(df)}")
        
        # Randomly sample records before processing
        if max_samples:
            # Estimate how many records we need to get desired number of samples
            # Assuming on average we get ~1 sample per window_size records
            obs_len = self.config['data']['obs_len']
            pred_len = self.config['data']['pred_len']
            window_size = obs_len + pred_len
            
            # We need more records than samples because not all records form valid windows
            # Use a multiplier (e.g., 3x) to ensure we get enough samples
            estimated_records_needed = max_samples * window_size * 3
            
            if len(df) > estimated_records_needed:
                print(f"Randomly sampling {estimated_records_needed} records from {len(df)} total records...")
                np.random.seed(42)  # Set seed for reproducibility
                sampled_indices = np.random.choice(len(df), size=estimated_records_needed, replace=False)
                df = df.iloc[sampled_indices].sort_values(['user_id', 'timestamp']).reset_index(drop=True)
                print(f"Sampled {len(df)} records for processing")
        
        # Group by user_id and create trajectory samples
        samples = []
        obs_len = self.config['data']['obs_len']
        pred_len = self.config['data']['pred_len']
        window_size = obs_len + pred_len
        
        for user_id, group in tqdm(df.groupby('user_id'), desc="Processing users"):
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            if len(group) < window_size:
                continue
            
            # Sliding window to create samples
            for i in range(len(group) - window_size + 1):
                window = group.iloc[i:i + window_size]
                
                # Extract observation and prediction
                obs = window.iloc[:obs_len]
                pred = window.iloc[obs_len:]
                
                # Build trajectory and next location
                trajectory = []
                for _, row in obs.iterrows():
                    trajectory.append({
                        'location_id': int(row['location_id']),
                        'timestamp': row['timestamp'],
                        'user_id': user_id
                    })
                
                next_location = int(pred.iloc[0]['location_id'])
                
                sample = {
                    'user_id': user_id,
                    'trajectory': trajectory,
                    'next_location': next_location,
                    'next_timestamp': pred.iloc[0]['timestamp']
                }
                
                samples.append(sample)
                
                # Early stopping if we have enough samples
                if max_samples and len(samples) >= max_samples:
                    break
            
            if max_samples and len(samples) >= max_samples:
                break
        
        print(f"Created {len(samples)} trajectory samples")
        
        # Final random shuffle to ensure diversity across users
        print("Randomly shuffling samples for diversity...")
        np.random.shuffle(samples)
        
        # Limit to max_samples if needed
        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]
            print(f"Selected {max_samples} samples")
        
        return samples
    
    def generate_embeddings(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for all trajectory samples.
        
        Args:
            samples: List of trajectory samples
        
        Returns:
            Numpy array of embeddings (shape: [n_samples, embedding_dim])
        """
        print("\n" + "="*80)
        print("Generating embeddings for all samples...")
        print("="*80)
        
        embeddings = []
        
        for sample in tqdm(samples, desc="Encoding trajectories"):
            trajectory = sample['trajectory']
            
            # Generate embedding using LLM
            embedding = self.llm.encode_trajectory(
                trajectory=trajectory,
                poi_data=self.poi_data,
                city=self.city
            )
            
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_components: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
        """
        Normalize, apply PCA, and cluster embeddings using HDBSCAN.
        
        Args:
            embeddings: Original embeddings (not normalized)
            n_components: Number of PCA components
        
        Returns:
            Tuple of (cluster_labels, normalized_embeddings, scaler, pca)
        """
        print("\n" + "="*80)
        print("Clustering embeddings...")
        print("="*80)
        
        # Step 1: Normalize embeddings (only for clustering)
        print("Normalizing embeddings...")
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings)
        print(f"Normalized embeddings shape: {normalized_embeddings.shape}")
        
        # Step 2: Apply PCA for dimensionality reduction
        print(f"Applying PCA (n_components={n_components})...")
        pca = PCA(n_components=min(n_components, normalized_embeddings.shape[0], normalized_embeddings.shape[1]))
        pca_embeddings = pca.fit_transform(normalized_embeddings)
        print(f"PCA embeddings shape: {pca_embeddings.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Step 3: HDBSCAN clustering
        print(f"Performing HDBSCAN clustering (target clusters: {self.target_clusters})...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, len(embeddings) // (self.target_clusters * 2)),
            min_samples=3,
            cluster_selection_epsilon=0.0,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(pca_embeddings)
        
        # Get cluster statistics
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        n_noise = np.sum(cluster_labels == -1)
        
        print(f"Number of clusters found: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        
        # Distribution of cluster sizes
        cluster_sizes = []
        for label in unique_labels:
            if label >= 0:
                cluster_sizes.append(np.sum(cluster_labels == label))
        
        if cluster_sizes:
            print(f"Cluster size statistics:")
            print(f"  Min: {np.min(cluster_sizes)}")
            print(f"  Max: {np.max(cluster_sizes)}")
            print(f"  Mean: {np.mean(cluster_sizes):.1f}")
            print(f"  Median: {np.median(cluster_sizes):.1f}")
        
        return cluster_labels, normalized_embeddings, scaler, pca
    
    def select_representative_samples(
        self,
        samples: List[Dict[str, Any]],
        embeddings: np.ndarray,
        normalized_embeddings: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, List[int]]:
        """
        Select representative samples from each cluster with functional coverage constraints.
        
        Args:
            samples: Original trajectory samples
            embeddings: Original embeddings (not normalized)
            normalized_embeddings: Normalized embeddings used for clustering
            cluster_labels: Cluster labels from HDBSCAN
        
        Returns:
            Tuple of (selected_samples, selected_embeddings, selected_indices)
        """
        print("\n" + "="*80)
        print("Selecting representative samples from each cluster...")
        print("="*80)
        
        selected_samples = []
        selected_embeddings = []
        selected_indices = []
        
        unique_labels = np.unique(cluster_labels)
        valid_clusters = unique_labels[unique_labels >= 0]
        
        print(f"Processing {len(valid_clusters)} clusters...")
        
        for cluster_id in tqdm(valid_clusters, desc="Selecting samples"):
            # Get all samples in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = normalized_embeddings[cluster_mask]
            
            if len(cluster_indices) == 0:
                continue
            
            # Compute cluster center
            cluster_center = cluster_embeddings.mean(axis=0)
            
            # Calculate distances to center
            distances_to_center = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            
            # Strategy 1: Select center prototype (closest to center)
            center_idx = cluster_indices[np.argmin(distances_to_center)]
            selected = [center_idx]
            
            # Strategy 2: Select edge samples (farthest from center)
            if len(cluster_indices) > 1:
                edge_idx = cluster_indices[np.argmax(distances_to_center)]
                if edge_idx not in selected:
                    selected.append(edge_idx)
            
            # Strategy 3: Select diverse samples (maximize coverage)
            remaining_indices = [idx for idx in cluster_indices if idx not in selected]
            
            while len(selected) < self.samples_per_cluster and remaining_indices:
                # Select sample that is most different from already selected
                best_idx = None
                best_score = -1
                
                for idx in remaining_indices:
                    # Calculate minimum distance to already selected samples
                    min_dist = min([
                        np.linalg.norm(normalized_embeddings[idx] - normalized_embeddings[sel_idx])
                        for sel_idx in selected
                    ])
                    
                    # Bonus for functional diversity
                    sample = samples[idx]
                    diversity_score = self._calculate_diversity_score(sample, [samples[s] for s in selected])
                    
                    # Combined score
                    score = min_dist + 0.5 * diversity_score
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx is not None:
                    selected.append(best_idx)
                    remaining_indices.remove(best_idx)
                else:
                    break
            
            # Add selected samples
            for idx in selected:
                selected_samples.append(samples[idx])
                selected_embeddings.append(embeddings[idx])  # Use original embedding
                selected_indices.append(idx)
        
        selected_embeddings = np.array(selected_embeddings)
        
        print(f"Selected {len(selected_samples)} representative samples")
        print(f"Average samples per cluster: {len(selected_samples) / len(valid_clusters):.1f}")
        
        return selected_samples, selected_embeddings, selected_indices
    
    def _calculate_diversity_score(
        self,
        sample: Dict[str, Any],
        selected_samples: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate functional diversity score for a sample.
        Considers origin function, time patterns, spatial coverage, and distance.
        
        Args:
            sample: Candidate sample
            selected_samples: Already selected samples
        
        Returns:
            Diversity score (higher = more diverse)
        """
        if not selected_samples:
            return 0.0
        
        trajectory = sample['trajectory']
        origin_grid = trajectory[0]['location_id']
        dest_grid = sample['next_location']
        
        # Parse timestamp for temporal features
        try:
            timestamp = trajectory[-1]['timestamp']
            if isinstance(timestamp, str):
                dt = datetime.strptime(timestamp, '%Y%m%d %H:%M')
                hour = dt.hour
                weekday = dt.weekday()
            else:
                hour = 12
                weekday = 3
        except:
            hour = 12
            weekday = 3
        
        # Calculate spatial distance
        try:
            origin_lat, origin_lon = grid_id_to_latlon(origin_grid, self.city)
            dest_lat, dest_lon = grid_id_to_latlon(dest_grid, self.city)
            distance = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)
        except:
            distance = 0.0
        
        # Get POI type of origin
        origin_poi_type = "unknown"
        if origin_grid in self.poi_data:
            poi_features = self.poi_data[origin_grid]
            max_poi = max(poi_features.items(), key=lambda x: x[1])
            if max_poi[1] > 10:  # Threshold for dominant POI
                origin_poi_type = max_poi[0]
        
        # Calculate diversity from selected samples
        diversity_scores = []
        
        for sel_sample in selected_samples:
            sel_trajectory = sel_sample['trajectory']
            sel_origin = sel_trajectory[0]['location_id']
            sel_dest = sel_sample['next_location']
            
            # Origin POI diversity
            sel_origin_poi = "unknown"
            if sel_origin in self.poi_data:
                poi_features = self.poi_data[sel_origin]
                max_poi = max(poi_features.items(), key=lambda x: x[1])
                if max_poi[1] > 10:
                    sel_origin_poi = max_poi[0]
            
            poi_diversity = 1.0 if origin_poi_type != sel_origin_poi else 0.0
            
            # Temporal diversity
            try:
                sel_timestamp = sel_trajectory[-1]['timestamp']
                if isinstance(sel_timestamp, str):
                    sel_dt = datetime.strptime(sel_timestamp, '%Y%m%d %H:%M')
                    sel_hour = sel_dt.hour
                    sel_weekday = sel_dt.weekday()
                else:
                    sel_hour = 12
                    sel_weekday = 3
            except:
                sel_hour = 12
                sel_weekday = 3
            
            hour_diff = min(abs(hour - sel_hour), 24 - abs(hour - sel_hour)) / 12.0
            weekday_diversity = 1.0 if weekday != sel_weekday else 0.0
            time_diversity = (hour_diff + weekday_diversity) / 2.0
            
            # Spatial diversity
            spatial_diversity = min(abs(origin_grid - sel_origin), abs(dest_grid - sel_dest)) / 1600.0
            
            # Distance diversity
            try:
                sel_origin_lat, sel_origin_lon = grid_id_to_latlon(sel_origin, self.city)
                sel_dest_lat, sel_dest_lon = grid_id_to_latlon(sel_dest, self.city)
                sel_distance = haversine_distance(sel_origin_lat, sel_origin_lon, sel_dest_lat, sel_dest_lon)
            except:
                sel_distance = 0.0
            
            distance_diversity = min(abs(distance - sel_distance), 10.0) / 10.0
            
            # Combined diversity
            sample_diversity = (
                poi_diversity * 0.3 +
                time_diversity * 0.3 +
                spatial_diversity * 0.2 +
                distance_diversity * 0.2
            )
            
            diversity_scores.append(sample_diversity)
        
        # Return average diversity
        return np.mean(diversity_scores)
    
    def save_database(
        self,
        samples: List[Dict[str, Any]],
        embeddings: np.ndarray
    ):
        """
        Save the RAG database to disk.
        
        Args:
            samples: Selected trajectory samples
            embeddings: Corresponding embeddings
        """
        print("\n" + "="*80)
        print("Saving RAG database...")
        print("="*80)
        
        # Save embeddings
        embeddings_path = os.path.join(self.output_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to: {embeddings_path}")
        print(f"  Shape: {embeddings.shape}")
        
        # Save samples metadata
        samples_path = os.path.join(self.output_dir, "samples.pkl")
        with open(samples_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"Saved samples to: {samples_path}")
        print(f"  Count: {len(samples)}")
        
        # Build and save FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Use IndexFlatIP for inner product (cosine similarity for normalized vectors)
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings_normalized.astype('float32'))
        
        faiss_index_path = os.path.join(self.output_dir, "faiss_index.bin")
        faiss.write_index(faiss_index, faiss_index_path)
        print(f"Saved FAISS index to: {faiss_index_path}")
        print(f"  Dimension: {dimension}")
        print(f"  Vectors: {faiss_index.ntotal}")
        
        # Save statistics
        stats = {
            'city': self.city,
            'n_samples': len(samples),
            'embedding_dim': embeddings.shape[1],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        stats_path = os.path.join(self.output_dir, "database_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to: {stats_path}")
        
        print("\n" + "="*80)
        print("RAG Database construction completed!")
        print("="*80)
    
    def build(self, train_path: str, max_train_samples: int = None):
        """
        Main pipeline to build the RAG database.
        
        Args:
            train_path: Path to train.csv
            max_train_samples: Maximum training samples to use (None = all)
        """
        # Step 1: Load training data
        samples = self.load_training_data(train_path, max_samples=max_train_samples)
        
        if len(samples) == 0:
            print("No samples loaded. Exiting.")
            return
        
        # Step 2: Generate embeddings
        embeddings = self.generate_embeddings(samples)
        
        # Step 3: Cluster embeddings
        cluster_labels, normalized_embeddings, scaler, pca = self.cluster_embeddings(embeddings)
        
        # Step 4: Select representative samples
        selected_samples, selected_embeddings, selected_indices = self.select_representative_samples(
            samples, embeddings, normalized_embeddings, cluster_labels
        )
        
        # Step 5: Save database
        self.save_database(selected_samples, selected_embeddings)


def main():
    """Main entry point for RAG database construction."""
    parser = argparse.ArgumentParser(description="Build RAG Experience Pool Database")
    parser.add_argument(
        '--config',
        type=str,
        default='/workspace/China_Journal/config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--city',
        type=str,
        default='beijing',
        help='City name (beijing, shenzhen, nanchang)'
    )
    parser.add_argument(
        '--target_clusters',
        type=int,
        default=1000,
        help='Target number of clusters'
    )
    parser.add_argument(
        '--samples_per_cluster',
        type=int,
        default=10,
        help='Number of samples per cluster'
    )
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=None,
        help='Maximum training samples to use (None = all)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen-3-8B',
        help='LLM model name for embedding generation'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration from:", args.config)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update city in config
    config['data']['city'] = args.city
    
    # Build train data path
    train_path = f"/workspace/China_Journal/data/{args.city}/train.csv"
    
    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        return
    
    # Create builder
    builder = RAGDatabaseBuilder(
        config=config,
        model_name=args.model_name,
        city=args.city,
        target_clusters=args.target_clusters,
        samples_per_cluster=args.samples_per_cluster
    )
    
    # Build database
    builder.build(train_path, max_train_samples=args.max_train_samples)


if __name__ == '__main__':
    main()
