"""
Improved Gravity Model Module for Candidate Location Selection
This module uses an improved gravity model to select candidate locations
based on POI distribution and distance from current location.

Formula: score_of_poi_A = weight * (num of category A in target grid) / distance^2
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import os
import json
from util.utils import calculate_grid_distance, grid_id_to_coordinates, coordinates_to_grid_id


class GravityModel:
    """
    Improved Gravity Model for mobility prediction.
    Calculates attraction scores for locations based on POI distribution and distance.
    """
    
    def __init__(
        self,
        poi_data_path: str,
        city: str = 'beijing',
        weight: Union[float, Dict[str, float]] = 1.0,
        gravity_top_n_candidates: int = 5,
        radius: int = 10,
        grid_size: int = 40,
        weight_config_path: Optional[str] = None
    ):
        """
        Initialize the Gravity Model.
        
        Args:
            poi_data_path: Path to POI data CSV file (e.g., beijing_grid_poi.csv)
            city: City name ('beijing', 'nanchang', 'shenzhen')
            weight: Weight parameter(s) in gravity model formula.
                   Can be a single float (same weight for all categories) or
                   a dictionary mapping POI category to weight value.
            gravity_top_n_candidates: Number of top candidate locations to return for each POI category
            radius: Search radius (in grid units) around current location
            grid_size: Size of the grid system (default 40x40 = 1600 grids)
            weight_config_path: Path to JSON file with fitted weights (optional)
        """
        self.city = city
        self.gravity_top_n_candidates = gravity_top_n_candidates
        self.radius = radius
        self.grid_size = grid_size
        
        # POI categories (14 categories) - with _count suffix
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
        
        # Load POI data
        self.poi_data = self._load_poi_data(poi_data_path)
        
        # Initialize weights
        self.weights = self._initialize_weights(weight, weight_config_path)
        # Initialize weights
        self.weights = self._initialize_weights(weight, weight_config_path)
    
    def _initialize_weights(
        self,
        weight: Union[float, Dict[str, float]],
        weight_config_path: Optional[str]
    ) -> Dict[str, float]:
        """
        Initialize weight parameters for each POI category.
        
        Priority:
        1. If weight_config_path provided, load from JSON file
        2. If weight is a dictionary, use it directly
        3. If weight is a float, use same weight for all categories
        
        Args:
            weight: Single weight or dictionary of weights
            weight_config_path: Path to weight config JSON file
        
        Returns:
            Dictionary mapping POI category to weight value
        """
        # Load from config file if provided
        if weight_config_path and os.path.exists(weight_config_path):
            print(f"Loading fitted weights from: {weight_config_path}")
            with open(weight_config_path, 'r') as f:
                config = json.load(f)
                if 'weights' in config:
                    return config['weights']
        
        # Use provided weights
        if isinstance(weight, dict):
            return weight
        else:
            # Use same weight for all categories
            return {cat: weight for cat in self.poi_categories}
    
    def _load_poi_data(self, poi_data_path: str) -> pd.DataFrame:
        """
        Load POI data from CSV file.
        
        Args:
            poi_data_path: Path to POI data file
        
        Returns:
            DataFrame with POI information
        """
        if not os.path.exists(poi_data_path):
            raise FileNotFoundError(f"POI data file not found: {poi_data_path}")
        
        df = pd.read_csv(poi_data_path)
        
        return df
    
    def _grid_id_to_coordinates(self, grid_id: int) -> Tuple[int, int]:
        """Use utility function for grid coordinate conversion."""
        return grid_id_to_coordinates(grid_id, self.grid_size)
    
    def _coordinates_to_grid_id(self, row: int, col: int) -> int:
        """Use utility function for grid ID conversion."""
        return coordinates_to_grid_id(row, col, self.grid_size)
    
    def _calculate_distance(self, grid_id_1: int, grid_id_2: int) -> float:
        """Use utility function for distance calculation with haversine formula."""
        return calculate_grid_distance(grid_id_1, grid_id_2, city=self.city, grid_size=self.grid_size)
    
    def _smooth_base_score(self, base_score: float) -> float:
        """
        Smooth base gravity score using logarithmic transformation.
        
        This prevents extreme differences in raw poi_count/distance^2 values
        from dominating the optimization. Uses log(1 + x) transformation.
        
        Args:
            base_score: Raw base score (poi_count / distance^2)
        
        Returns:
            Smoothed base score
        """
        if base_score <= 0:
            return 0.0
        # Use log(1 + x) to smooth the distribution
        return np.log1p(base_score)
    
    def _smooth_score(self, score: float, scores_list: List[float]) -> float:
        """
        Normalize gravity score to [0, 1] range.
        
        Uses min-max normalization across all scores in the candidate set.
        This is the final step after applying weights to smoothed base scores.
        
        Args:
            score: Weighted gravity score
            scores_list: List of all weighted scores for normalization
        
        Returns:
            Normalized score in [0, 1] range, rounded to 3 decimal places
        """
        if len(scores_list) == 0 or max(scores_list) == min(scores_list):
            return 0.0
        
        # Min-max normalization
        normalized = (score - min(scores_list)) / (max(scores_list) - min(scores_list))
        
        # Round to 3 decimal places
        return round(normalized, 3)
    
    def _get_grids_within_radius(self, current_grid_id: int) -> List[int]:
        """
        Get all grid IDs within the specified radius of current location.
        Uses circular radius (Euclidean distance).
        
        Args:
            current_grid_id: Current location grid ID
        
        Returns:
            List of grid IDs within radius
        """
        current_row, current_col = self._grid_id_to_coordinates(current_grid_id)
        
        grids_within_radius = []
        
        # Search in a square bounding box first, then filter by circular radius
        for row in range(max(0, current_row - self.radius), 
                        min(self.grid_size, current_row + self.radius + 1)):
            for col in range(max(0, current_col - self.radius), 
                            min(self.grid_size, current_col + self.radius + 1)):
                grid_id = self._coordinates_to_grid_id(row, col)
                
                # Check if within circular radius
                # distance = self._calculate_distance(current_grid_id, grid_id)
                # if distance <= self.radius:
                grids_within_radius.append(grid_id)
        
        return grids_within_radius
    
    def _calculate_gravity_score(
        self,
        current_grid_id: int,
        target_grid_id: int,
        poi_category: str
    ) -> float:
        """
        Calculate gravity score for a target grid using the improved gravity model.
        
        Process:
        1. Calculate base score: poi_count / distance^2
        2. Apply logarithmic smoothing to base score
        3. Multiply by category-specific weight
        
        Special handling for staying in place: uses a fixed moderate base score.
        
        Args:
            current_grid_id: Current location grid ID
            target_grid_id: Target grid ID
            poi_category: POI category name
        
        Returns:
            Weighted gravity score (after smoothing, before final normalization)
        """
        # Get weight for this POI category
        weight = self.weights.get(poi_category, 1.0)
        
        # Special case: staying in the same place
        if current_grid_id == target_grid_id:
            # Use a fixed moderate base score
            # After log smoothing: log(1 + 10) ≈ 2.4
            base_score = 100
            smoothed_base = self._smooth_base_score(base_score)
            return weight * smoothed_base
        
        # Get POI data for the target grid
        poi_row = self.poi_data[self.poi_data['grid_id'] == target_grid_id]
        
        if poi_row.empty:
            return 0.0
        
        # Get the count for the specific POI category
        if poi_category in poi_row.columns:
            poi_count = poi_row[poi_category].values[0]
        else:
            poi_count = 0
        
        if poi_count == 0:
            return 0.0
        
        # Calculate distance
        distance = self._calculate_distance(current_grid_id, target_grid_id)
        
        # Avoid division by zero - set minimum distance
        if distance < 0.01:  # Less than 10 meters
            distance = 0.01
        
        # Step 1: Calculate base gravity score
        base_score = poi_count / (distance ** 2)
        
        # Step 2: Apply logarithmic smoothing to avoid extreme values
        smoothed_base = self._smooth_base_score(base_score)
        
        # Step 3: Apply category-specific weight
        score = weight * smoothed_base
        
        return score
    
    def get_candidate_locations(
        self,
        current_grid_id: int,
        return_scores: bool = False
    ) -> Dict[str, List]:
        """
        Get candidate locations for each POI category using the gravity model.
        
        For each POI category, calculates gravity scores for all grids within radius,
        applies smoothing to normalize scores to [0, 1], then returns the top-n grids.
        Always includes the current location as a candidate (for stationary behavior).
        
        Args:
            current_grid_id: Current location grid ID
            return_scores: If True, return tuples of (grid_id, smoothed_score); 
                          If False, return only grid_ids
        
        Returns:
            Dictionary mapping POI category to list of top-n candidates
            - If return_scores=False: {category: [grid_id1, grid_id2, ...]}
            - If return_scores=True: {category: [(grid_id1, score1), (grid_id2, score2), ...]}
              where scores are smoothed to [0, 1] range
        """
        # Get all grids within the specified radius
        candidate_grids = self._get_grids_within_radius(current_grid_id)
        
        # Dictionary to store candidates for each POI category
        candidates = {}
        
        # For each POI category, calculate scores and select top-n
        for poi_category in self.poi_categories:
            scores = []
            raw_scores_list = []
            
            # Calculate raw scores for all candidates
            for target_grid_id in candidate_grids:
                score = self._calculate_gravity_score(
                    current_grid_id,
                    target_grid_id,
                    poi_category
                )
                scores.append((target_grid_id, score))
                raw_scores_list.append(score)
            
            # Apply smoothing to normalize scores to [0, 1]
            smoothed_scores = []
            for grid_id, raw_score in scores:
                smoothed_score = self._smooth_score(raw_score, raw_scores_list)
                smoothed_scores.append((grid_id, smoothed_score))
            
            # Sort by smoothed score (descending)
            smoothed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Always ensure current location is included
            top_candidates = []
            current_included = False
            
            for grid_id, score in smoothed_scores:
                if grid_id == current_grid_id:
                    current_included = True
                top_candidates.append((grid_id, score))
                if len(top_candidates) >= self.gravity_top_n_candidates:
                    break
            
            # If current location wasn't in top-n, add it
            if not current_included:
                current_raw_score = self._calculate_gravity_score(
                    current_grid_id, current_grid_id, poi_category
                )
                current_smoothed_score = self._smooth_score(current_raw_score, raw_scores_list)
                top_candidates.append((current_grid_id, current_smoothed_score))
            
            if return_scores:
                candidates[poi_category] = top_candidates
            else:
                candidates[poi_category] = [grid_id for grid_id, _ in top_candidates]
        
        return candidates
    
    def get_all_candidate_locations(
        self,
        current_grid_id: int,
        deduplicate: bool = True
    ) -> List[int]:
        """
        Get all candidate locations across all POI categories.
        
        Combines candidates from all POI categories into a single list.
        
        Args:
            current_grid_id: Current location grid ID
            deduplicate: If True, remove duplicate grid IDs
        
        Returns:
            List of candidate grid IDs
        """
        candidates_by_category = self.get_candidate_locations(current_grid_id)
        
        all_candidates = []
        for category, grid_ids in candidates_by_category.items():
            all_candidates.extend(grid_ids)
        
        if deduplicate:
            all_candidates = list(set(all_candidates))
        
        return all_candidates
    
    def get_top_k_candidates(
        self,
        current_grid_id: int,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get top-k candidate locations with highest average scores across all POI categories.
        
        This provides an overall ranking of candidate locations by averaging their
        gravity scores across all POI categories.
        
        Args:
            current_grid_id: Current location grid ID
            k: Number of top candidates to return
        
        Returns:
            List of tuples (grid_id, average_score) sorted by score (descending)
        """
        # Get all grids within radius
        candidate_grids = self._get_grids_within_radius(current_grid_id)
        
        # Calculate average score across all POI categories for each grid
        grid_scores = {}
        
        for target_grid_id in candidate_grids:
            scores = []
            for poi_category in self.poi_categories:
                score = self._calculate_gravity_score(
                    current_grid_id,
                    target_grid_id,
                    poi_category
                )
                scores.append(score)
            
            # Calculate average score
            avg_score = np.mean(scores)
            grid_scores[target_grid_id] = avg_score
        
        # Sort by average score and select top-k
        sorted_candidates = sorted(
            grid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return sorted_candidates
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the gravity model configuration.
        
        Returns:
            Dictionary with configuration statistics
        """
        return {
            'city': self.city,
            'weights': self.weights,
            'gravity_top_n_candidates': self.gravity_top_n_candidates,
            'radius': self.radius,
            'grid_size': self.grid_size,
            'total_grids': self.grid_size ** 2,
            'poi_categories': len(self.poi_categories),
            'poi_locations': len(self.poi_data)
        }
