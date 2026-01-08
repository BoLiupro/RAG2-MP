"""
Gravity Model Weight Parameter Fitting Script
This script fits optimal weight parameters for each POI category in the gravity model
based on training data to minimize prediction error.

Formula: score = weight_of_category * POI_count / distance^2

The script will:
1. Load training trajectory data
2. For each POI category, fit an optimal weight parameter
3. Save the fitted weights for use in the Gravity Model
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import sys
from scipy.optimize import minimize
from tqdm import tqdm
import json
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.utils import calculate_grid_distance, grid_id_to_coordinates, CITY_BOUNDARIES


class GravityWeightFitter:
    """
    Fits optimal weight parameters for the gravity model based on training data.
    """
    
    def __init__(
        self,
        city: str,
        train_data_path: str,
        poi_data_path: str,
        obs_len: int = 12,
        pred_len: int = 1,
        radius: int = 10,
        grid_size: int = 40,
        use_gpu: bool = True
    ):
        """
        Initialize the weight fitter.
        
        Args:
            city: City name ('beijing', 'nanchang', 'shenzhen')
            train_data_path: Path to training data CSV
            poi_data_path: Path to POI data CSV
            obs_len: Observation length (number of historical locations)
            pred_len: Prediction length (should be 1)
            radius: Search radius for candidate grids
            grid_size: Grid system size (40x40)
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        self.city = city
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.radius = radius
        self.grid_size = grid_size
        
        # GPU setup
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        if self.use_gpu:
            print(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        else:
            print(f"✗ GPU not available, using CPU")
        
        print(f"Loading data for {city}...")
        self.train_data = pd.read_csv(train_data_path)
        self.poi_data = pd.read_csv(poi_data_path)
        
        # POI categories (14 categories)
        self.poi_categories = [
            'Transportation Facilities_count',
            'Leisure & Entertainment_count',
            'Companies & Enterprises_count',
            'Healthcare_count',
            'Residential_count',
            'Tourist Attractions_count',
            'Automotive_count',
            'Life Services_count',
            'Science & Education & Culture_count',
            'Shopping & Consumer Goods_count',
            'Sports & Fitness_count',
            'Hotels & Accommodations_count',
            'Financial Institutions_count',
            'Dining & Cuisine_count'
        ]
        
        print(f"Loaded {len(self.train_data)} training records")
        print(f"Loaded {len(self.poi_data)} POI grids")
        
        # Create trajectory samples
        self.samples = self._create_samples()
        print(f"Created {len(self.samples)} trajectory samples")
        
        # Precompute grid coordinates for GPU
        if self.use_gpu:
            self._precompute_gpu_data()
    
    def _create_samples(self) -> List[Dict]:
        """
        Create trajectory samples from training data.
        Each sample contains obs_len historical locations and pred_len future location(s).
        
        Filters out samples where the target location is the same as the current location
        (staying in place), as these don't help fit meaningful weights.
        
        Returns:
            List of sample dictionaries with 'obs' and 'target' keys
        """
        samples = []
        filtered_count = 0
        
        # Group by user
        grouped = self.train_data.groupby('user_id')
        
        for user_id, user_df in grouped:
            user_df = user_df.sort_values('timestamp').reset_index(drop=True)
            locations = user_df['location_id'].values
            
            # Create sliding windows
            for i in range(len(locations) - self.obs_len - self.pred_len + 1):
                obs = locations[i:i + self.obs_len]
                target = locations[i + self.obs_len:i + self.obs_len + self.pred_len]
                
                # Filter out samples where user stays in the same place
                current_location = obs[-1]
                next_location = target[0]
                
                if current_location != next_location:
                    samples.append({
                        'obs': obs,
                        'target': next_location
                    })
                else:
                    filtered_count += 1
        
        print(f"Filtered out {filtered_count} samples where user stayed in place")
        return samples
    
    def _precompute_gpu_data(self):
        """
        Precompute and cache data on GPU for faster computation.
        """
        print("Precomputing GPU data...")
        
        # Get city boundaries
        city_info = CITY_BOUNDARIES[self.city]
        lat_min, lat_max = city_info['lat_min'], city_info['lat_max']
        lon_min, lon_max = city_info['lon_min'], city_info['lon_max']
        
        # Precompute grid center coordinates for all grids
        grid_lats = []
        grid_lons = []
        for grid_id in range(self.grid_size * self.grid_size):
            row, col = grid_id_to_coordinates(grid_id, self.grid_size)
            lat = lat_min + (row + 0.5) * (lat_max - lat_min) / self.grid_size
            lon = lon_min + (col + 0.5) * (lon_max - lon_min) / self.grid_size
            grid_lats.append(lat)
            grid_lons.append(lon)
        
        self.grid_lats_gpu = torch.tensor(grid_lats, dtype=torch.float32, device=self.device)
        self.grid_lons_gpu = torch.tensor(grid_lons, dtype=torch.float32, device=self.device)
        
        # Precompute POI counts for all grids and categories
        self.poi_counts_gpu = {}
        for poi_category in self.poi_categories:
            poi_counts = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
            for _, row in self.poi_data.iterrows():
                grid_id = int(row['grid_id'])
                if poi_category in row:
                    poi_counts[grid_id] = row[poi_category]
            self.poi_counts_gpu[poi_category] = torch.tensor(
                poi_counts, dtype=torch.float32, device=self.device
            )
        
        print(f"✓ GPU data precomputed: {self.grid_size*self.grid_size} grids, {len(self.poi_categories)} categories")
    
    def _haversine_distance_gpu(self, lat1: torch.Tensor, lon1: torch.Tensor, 
                                lat2: torch.Tensor, lon2: torch.Tensor) -> torch.Tensor:
        """
        Calculate haversine distance on GPU (vectorized).
        
        Args:
            lat1, lon1: First point coordinates (can be scalars or tensors)
            lat2, lon2: Second point coordinates (can be scalars or tensors)
        
        Returns:
            Distance in kilometers (tensor)
        """
        R = 6371.0  # Earth radius in km
        
        lat1_rad = torch.deg2rad(lat1)
        lat2_rad = torch.deg2rad(lat2)
        lon1_rad = torch.deg2rad(lon1)
        lon2_rad = torch.deg2rad(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
        c = 2 * torch.arcsin(torch.sqrt(a))
        
        return R * c
    
    def _calculate_gravity_scores_batch_gpu(
        self,
        current_grid_ids: torch.Tensor,
        candidate_grid_ids: torch.Tensor,
        poi_category: str,
        weight: float
    ) -> torch.Tensor:
        """
        Calculate gravity scores for multiple current-candidate pairs on GPU (vectorized).
        
        Args:
            current_grid_ids: Tensor of current grid IDs [batch_size]
            candidate_grid_ids: Tensor of candidate grid IDs [batch_size, num_candidates]
            poi_category: POI category name
            weight: Weight parameter
        
        Returns:
            Scores tensor [batch_size, num_candidates]
        """
        # Convert weight to tensor if it's a numpy array
        if isinstance(weight, np.ndarray):
            weight = torch.tensor(weight.item(), dtype=torch.float32, device=self.device)
        elif not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float32, device=self.device)
        
        batch_size, num_candidates = candidate_grid_ids.shape
        
        # Get coordinates for current grids
        current_lats = self.grid_lats_gpu[current_grid_ids]  # [batch_size]
        current_lons = self.grid_lons_gpu[current_grid_ids]
        
        # Get coordinates for candidate grids
        candidate_lats = self.grid_lats_gpu[candidate_grid_ids]  # [batch_size, num_candidates]
        candidate_lons = self.grid_lons_gpu[candidate_grid_ids]
        
        # Calculate distances (vectorized)
        current_lats_expanded = current_lats.unsqueeze(1).expand_as(candidate_lats)
        current_lons_expanded = current_lons.unsqueeze(1).expand_as(candidate_lons)
        
        distances = self._haversine_distance_gpu(
            current_lats_expanded, current_lons_expanded,
            candidate_lats, candidate_lons
        )
        
        # Get POI counts
        poi_counts = self.poi_counts_gpu[poi_category][candidate_grid_ids]
        
        # Special handling for same location
        same_location_mask = (current_grid_ids.unsqueeze(1) == candidate_grid_ids)
        
        # Calculate scores with new 3-step process:
        # Step 1: Calculate base score = poi_count / distance^2
        distances = torch.clamp(distances, min=0.01)  # Avoid division by zero
        base_scores = poi_counts / (distances ** weight)
        
        # Step 2: Apply logarithmic smoothing
        smoothed_base_scores = torch.log1p(base_scores)
        
        # Step 3: Apply weight
        scores = smoothed_base_scores
        
        # Apply fixed smoothed score for same location (base=10, log(11)≈2.4)
        same_location_smoothed = weight * torch.log1p(torch.tensor(10.0, device=self.device))
        scores[same_location_mask] = same_location_smoothed
        
        return scores
    
    def _get_grids_within_radius(self, current_grid_id: int) -> List[int]:
        """
        Get all grid IDs within the specified radius.
        
        Args:
            current_grid_id: Current location grid ID
        
        Returns:
            List of grid IDs within radius
        """
        current_row, current_col = grid_id_to_coordinates(current_grid_id, self.grid_size)
        
        grids_within_radius = []
        
        for row in range(max(0, current_row - self.radius), 
                        min(self.grid_size, current_row + self.radius + 1)):
            for col in range(max(0, current_col - self.radius), 
                            min(self.grid_size, current_col + self.radius + 1)):
                grid_id = current_row * self.grid_size + col
                grids_within_radius.append(grid_id)
        
        return grids_within_radius
    
    def _smooth_base_score(self, base_score: float) -> float:
        """
        Smooth base gravity score using logarithmic transformation.
        Same as in Gravity.py model.
        """
        if base_score <= 0:
            return 0.0
        return np.log1p(base_score)
    
    def _calculate_gravity_score(
        self,
        current_grid_id: int,
        target_grid_id: int,
        poi_category: str,
        weight: float
    ) -> float:
        """
        Calculate gravity score for a target grid.
        
        Process:
        1. Calculate base score: poi_count / distance^2
        2. Apply logarithmic smoothing to base score  
        3. Multiply by weight
        
        Args:
            current_grid_id: Current location grid ID
            target_grid_id: Target grid ID
            poi_category: POI category name
            weight: Weight parameter
        
        Returns:
            Weighted gravity score
        """
        # Special case: staying in the same place
        if current_grid_id == target_grid_id:
            base_score = 10.0
            smoothed_base = self._smooth_base_score(base_score)
            return weight * smoothed_base
        
        # Get POI count for the target grid
        poi_row = self.poi_data[self.poi_data['grid_id'] == target_grid_id]
        
        if poi_row.empty:
            poi_count = 0
        else:
            if poi_category in poi_row.columns:
                poi_count = poi_row[poi_category].values[0]
            else:
                poi_count = 0
        
        if poi_count == 0:
            return 0.0
        
        # Calculate distance
        distance = calculate_grid_distance(
            current_grid_id, target_grid_id,
            city=self.city, grid_size=self.grid_size
        )
        
        # Avoid division by zero for very close grids
        if distance < 0.01:  # Less than 10 meters
            distance = 0.01
        
        # Step 1: Calculate base gravity score
        base_score = poi_count / (distance ** 2)
        
        # Step 2: Apply logarithmic smoothing
        smoothed_base = self._smooth_base_score(base_score)
        
        # Step 3: Apply weight
        score = weight * smoothed_base
        
        return score
    
    def _objective_function(self, weight: float, poi_category: str, samples: List[Dict]) -> float:
        """
        Objective function to minimize: average ranking of the true next location.
        
        Lower is better - ideally the true location should be ranked #1.
        Uses GPU acceleration for batch processing if available.
        
        Args:
            weight: Weight parameter to optimize
            poi_category: POI category
            samples: List of trajectory samples
        
        Returns:
            Average ranking of true next location (lower is better)
        """
        if self.use_gpu and len(samples) > 50:
            # Use GPU batch processing for larger sample sets
            return self._objective_function_gpu_batch(weight, poi_category, samples)
        else:
            # Use CPU for small sample sets
            return self._objective_function_cpu(weight, poi_category, samples)
    
    def _objective_function_cpu(self, weight: float, poi_category: str, samples: List[Dict]) -> float:
        """CPU version of objective function (original implementation)."""
        rankings = []
        
        for sample in samples:
            current_grid_id = sample['obs'][-1]
            true_next_grid = sample['target']
            
            # Get candidate grids
            candidate_grids = self._get_grids_within_radius(current_grid_id)
            
            # Calculate scores for all candidates
            scores = []
            for grid_id in candidate_grids:
                score = self._calculate_gravity_score(
                    current_grid_id, grid_id, poi_category, weight
                )
                scores.append((grid_id, score))
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Find ranking of true next location
            rank = None
            for idx, (grid_id, _) in enumerate(scores):
                if grid_id == true_next_grid:
                    rank = idx + 1  # 1-indexed
                    break
            
            if rank is not None:
                rankings.append(rank)
        
        # Return average ranking (lower is better)
        if len(rankings) == 0:
            return 1000.0  # Large penalty if no matches
        
        return np.mean(rankings)
    
    def _objective_function_gpu_batch(self, weight: float, poi_category: str, samples: List[Dict]) -> float:
        """
        GPU-accelerated batch version of objective function.
        Processes multiple samples in parallel on GPU.
        """
        batch_size = 128  # Process 128 samples at a time
        all_rankings = []
        
        # Process samples in batches
        for batch_start in range(0, len(samples), batch_size):
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            
            # Prepare batch data
            current_grids = []
            true_next_grids = []
            all_candidates = []
            max_candidates = 0
            
            for sample in batch_samples:
                current_grid = sample['obs'][-1]
                true_next = sample['target']
                candidates = self._get_grids_within_radius(current_grid)
                
                current_grids.append(current_grid)
                true_next_grids.append(true_next)
                all_candidates.append(candidates)
                max_candidates = max(max_candidates, len(candidates))
            
            # Pad candidates to same length
            candidate_matrix = []
            for candidates in all_candidates:
                padded = candidates + [candidates[0]] * (max_candidates - len(candidates))
                candidate_matrix.append(padded)
            
            # Convert to tensors
            current_grids_t = torch.tensor(current_grids, dtype=torch.long, device=self.device)
            candidate_matrix_t = torch.tensor(candidate_matrix, dtype=torch.long, device=self.device)
            true_next_grids_t = torch.tensor(true_next_grids, dtype=torch.long, device=self.device)
            
            # Calculate scores for all candidates (vectorized)
            scores = self._calculate_gravity_scores_batch_gpu(
                current_grids_t, candidate_matrix_t, poi_category, weight
            )  # [batch_size, max_candidates]
            
            # Find rankings
            for i in range(len(batch_samples)):
                num_real_candidates = len(all_candidates[i])
                sample_scores = scores[i, :num_real_candidates]
                sample_candidates = candidate_matrix_t[i, :num_real_candidates]
                true_next = true_next_grids_t[i]
                
                # Sort by score descending
                sorted_indices = torch.argsort(sample_scores, descending=True)
                sorted_candidates = sample_candidates[sorted_indices]
                
                # Find rank of true location
                matches = (sorted_candidates == true_next).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    rank = matches[0].item() + 1  # 1-indexed
                    all_rankings.append(rank)
        
        # Return average ranking
        if len(all_rankings) == 0:
            return 1000.0
        
        return np.mean(all_rankings)
    
    def fit_weights(
        self,
        max_samples_per_category: int = 1000,
        initial_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Fit optimal weight parameters for each POI category.
        
        Args:
            max_samples_per_category: Maximum samples to use per category (for efficiency)
            initial_weight: Initial weight value for optimization
        
        Returns:
            Dictionary mapping POI category to optimal weight
        """
        optimal_weights = {}
        
        print("\n" + "="*80)
        print("FITTING GRAVITY MODEL WEIGHTS")
        print("="*80)
        
        # Sample data if too large
        if len(self.samples) > max_samples_per_category:
            sample_indices = np.random.choice(
                len(self.samples),
                size=max_samples_per_category,
                replace=False
            )
            samples_to_use = [self.samples[i] for i in sample_indices]
            print(f"\nUsing {max_samples_per_category} samples for fitting (randomly sampled)")
        else:
            samples_to_use = self.samples
            print(f"\nUsing all {len(samples_to_use)} samples for fitting")
        
        # Fit weight for each POI category
        for idx, poi_category in enumerate(self.poi_categories, 1):
            category_name = poi_category.replace('_count', '')
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(self.poi_categories)}] Fitting: {category_name}")
            print(f"{'='*60}")
            
            # Track optimization iterations
            iteration_count = [0]
            best_score = [float('inf')]
            
            def callback_func(xk):
                """Callback to show optimization progress"""
                iteration_count[0] += 1
                current_score = self._objective_function(xk[0] if hasattr(xk, '__len__') else xk, 
                                                         poi_category, samples_to_use)
                if current_score < best_score[0]:
                    best_score[0] = current_score
                
                # Update progress every iteration
                print(f"  Iter {iteration_count[0]:2d}: weight={xk[0] if hasattr(xk, '__len__') else xk:.4f}, "
                      f"avg_rank={current_score:.2f} {'✓' if current_score < best_score[0] + 0.01 else ''}", 
                      end='\r')
            
            # Optimize weight parameter
            result = minimize(
                fun=lambda w: self._objective_function(w, poi_category, samples_to_use),
                x0=[initial_weight],
                method='Nelder-Mead',
                options={'maxiter': 50, 'disp': False},
                callback=callback_func
            )
            
            optimal_weight = max(0.01, result.x[0])  # Ensure positive weight
            optimal_weights[poi_category] = optimal_weight
            
            print(f"\n  ✓ Optimal weight: {optimal_weight:.4f}")
            print(f"  ✓ Final avg ranking: {result.fun:.2f}")
            print(f"  ✓ Iterations: {result.nit}")
        
        return optimal_weights
    
    def evaluate_weights(
        self,
        weights: Dict[str, float],
        max_eval_samples: int = 500
    ) -> Dict[str, float]:
        """
        Evaluate the fitted weights on training data.
        
        Args:
            weights: Dictionary of fitted weights
            max_eval_samples: Maximum samples for evaluation
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*80)
        print("EVALUATING FITTED WEIGHTS")
        print("="*80)
        
        # Sample data if too large
        if len(self.samples) > max_eval_samples:
            sample_indices = np.random.choice(
                len(self.samples),
                size=max_eval_samples,
                replace=False
            )
            eval_samples = [self.samples[i] for i in sample_indices]
        else:
            eval_samples = self.samples
        
        # Evaluate each category
        results = {}
        
        for poi_category in self.poi_categories:
            weight = weights[poi_category]
            avg_rank = self._objective_function(weight, poi_category, eval_samples)
            results[poi_category] = avg_rank
            
            print(f"{poi_category.replace('_count', '')}: avg_rank={avg_rank:.2f}")
        
        # Overall statistics
        avg_overall = np.mean(list(results.values()))
        print(f"\n{'='*60}")
        print(f"Overall average ranking: {avg_overall:.2f}")
        print(f"{'='*60}")
        
        return results


def main():
    """
    Main function to fit gravity model weights.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Fit gravity model weights')
    parser.add_argument('--city', type=str, required=True,
                       choices=['beijing', 'nanchang', 'shenzhen'],
                       help='City name')
    parser.add_argument('--data_dir', type=str,
                       default='/workspace/China_Journal/data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/China_Journal/util/gravity_weight',
                       help='Output directory for fitted weights')
    parser.add_argument('--obs_len', type=int, default=12,
                       help='Observation length')
    parser.add_argument('--radius', type=int, default=10,
                       help='Search radius')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples for fitting')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration (use CPU only)')
    
    args = parser.parse_args()
    
    # Paths
    train_data_path = os.path.join(args.data_dir, args.city, 'train.csv')
    poi_data_path = os.path.join(args.data_dir, args.city, 'poi.csv')
    
    # Create fitter
    fitter = GravityWeightFitter(
        city=args.city,
        train_data_path=train_data_path,
        poi_data_path=poi_data_path,
        obs_len=args.obs_len,
        radius=args.radius,
        use_gpu=not args.no_gpu
    )
    
    # Fit weights
    optimal_weights = fitter.fit_weights(max_samples_per_category=args.max_samples)
    
    # Evaluate weights
    eval_results = fitter.evaluate_weights(optimal_weights)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    output_path = os.path.join(args.output_dir, f'gravity_weights_{args.city}.json')
    
    results = {
        'city': args.city,
        'weights': optimal_weights,
        'evaluation': eval_results,
        'metadata': {
            'obs_len': args.obs_len,
            'radius': args.radius,
            'num_samples': len(fitter.samples),
            'num_training_records': len(fitter.train_data)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF FITTED WEIGHTS")
    print("="*80)
    for category, weight in optimal_weights.items():
        print(f"{category.replace('_count', ''):<40} : {weight:.4f}")


if __name__ == '__main__':
    main()
