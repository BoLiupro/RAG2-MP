"""
Improved Gravity Model Module for Candidate Location Selection
This module uses an improved gravity model to select candidate locations
based on POI distribution and distance from current location.

Formula: score_of_poi_A = weight * (num of category A in target grid) / distance^2
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
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
        weight: float = 1.0,
        gravity_top_n_candidates: int = 5,
        radius: int = 10,
        grid_size: int = 40
    ):
        """
        Initialize the Gravity Model.
        
        Args:
            poi_data_path: Path to POI data CSV file (e.g., beijing_grid_poi.csv)
            city: City name ('beijing', 'nanchang', 'shenzhen')
            weight: Weight parameter in gravity model formula
            gravity_top_n_candidates: Number of top candidate locations to return for each POI category
            radius: Search radius (in grid units) around current location
            grid_size: Size of the grid system (default 40x40 = 1600 grids)
        """
        self.city = city
        self.weight = weight
        self.gravity_top_n_candidates = gravity_top_n_candidates
        self.radius = radius
        self.grid_size = grid_size
        
        # Load POI data
        self.poi_data = self._load_poi_data(poi_data_path)
        
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
            'Dining & Cusine_count'  # Note: 'Cusine' is spelled this way in the data
        ]
        
        self.poi_data = self._load_poi_data(poi_data_path)
    
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
        
        Formula: score = weight * (num_of_category_A_in_target_grid) / distance^2
        
        Args:
            current_grid_id: Current location grid ID
            target_grid_id: Target grid ID
            poi_category: POI category name
        
        Returns:
            Gravity score
        """
        # Get POI data for the target grid
        poi_row = self.poi_data[self.poi_data['grid_id'] == target_grid_id]
        
        if poi_row.empty:
            return 0.0
        
        # Get the count for the specific POI category
        if poi_category in poi_row.columns:
            poi_count = poi_row[poi_category].values[0]
        else:
            poi_count = 0
        
        # Calculate distance
        distance = self._calculate_distance(current_grid_id, target_grid_id)
        
        # Calculate gravity score using the improved formula
        score = self.weight * poi_count / (distance ** 2)
        
        return score
    
    def get_candidate_locations(
        self,
        current_grid_id: int,
        return_scores: bool = False,
        include_current: bool = True
    ) -> Dict[str, List]:
        """
        Get candidate locations for each POI category using the gravity model.
        
        For each POI category, calculates gravity scores for all grids within radius,
        then returns the top-n grids with highest scores.
        Optionally includes the current location as a candidate (for stationary behavior).
        
        Args:
            current_grid_id: Current location grid ID
            return_scores: If True, return tuples of (grid_id, score); 
                          If False, return only grid_ids
            include_current: If True, always include current location as a candidate
        
        Returns:
            Dictionary mapping POI category to list of top-n candidates
            - If return_scores=False: {category: [grid_id1, grid_id2, ...]}
            - If return_scores=True: {category: [(grid_id1, score1), (grid_id2, score2), ...]}
        """
        # Get all grids within the specified radius
        candidate_grids = self._get_grids_within_radius(current_grid_id)
        
        # Dictionary to store candidates for each POI category
        candidates = {}
        
        # For each POI category, calculate scores and select top-n
        for poi_category in self.poi_categories:
            scores = []
            
            for target_grid_id in candidate_grids:
                score = self._calculate_gravity_score(
                    current_grid_id,
                    target_grid_id,
                    poi_category
                )
                scores.append((target_grid_id, score))
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top-n candidates
            if include_current:
                # Ensure current location is included
                top_candidates = []
                current_included = False
                
                for grid_id, score in scores:
                    if grid_id == current_grid_id:
                        current_included = True
                    top_candidates.append((grid_id, score))
                    if len(top_candidates) >= self.gravity_top_n_candidates:
                        break
                
                # If current location wasn't in top-n, add it
                if not current_included:
                    current_score = self._calculate_gravity_score(
                        current_grid_id, current_grid_id, poi_category
                    )
                    top_candidates.append((current_grid_id, current_score))
            else:
                top_candidates = scores[:self.gravity_top_n_candidates]
            
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
    
    def print_candidates(self, current_grid_id: int):
        """
        Print candidate locations for each POI category in a formatted way.
        
        Args:
            current_grid_id: Current location grid ID
        """
        # Display method does not print in model code
        # This method is kept for backwards compatibility but does nothing
        pass
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the gravity model configuration.
        
        Returns:
            Dictionary with configuration statistics
        """
        return {
            'city': self.city,
            'weight': self.weight,
            'gravity_top_n_candidates': self.gravity_top_n_candidates,
            'radius': self.radius,
            'grid_size': self.grid_size,
            'total_grids': self.grid_size ** 2,
            'poi_categories': len(self.poi_categories),
            'poi_locations': len(self.poi_data)
        }
    
    def print_statistics(self):
        """Print gravity model statistics."""
        # Print method does not print in model code
        # This method is kept for backwards compatibility but does nothing
        pass
