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
        top_n: int = 5,
        radius: int = 10,
        grid_size: int = 40
    ):
        """
        Initialize the Gravity Model.
        
        Args:
            poi_data_path: Path to POI data CSV file (e.g., beijing_grid_poi.csv)
            city: City name ('beijing', 'nanchang', 'shenzhen')
            weight: Weight parameter in gravity model formula
            top_n: Number of top candidate locations to return for each POI category
            radius: Search radius (in grid units) around current location
            grid_size: Size of the grid system (default 40x40 = 1600 grids)
        """
        self.city = city
        self.weight = weight
        self.top_n = top_n
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
        
        print(f"Initialized GravityModel for {city}")
        print(f"Parameters: weight={weight}, top_n={top_n}, radius={radius}")
        print(f"Loaded POI data for {len(self.poi_data)} locations")
    
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
        print(f"Loaded POI data from {poi_data_path}")
        
        return df
    
    def _grid_id_to_coordinates(self, grid_id: int) -> Tuple[int, int]:
        """
        Convert grid ID to grid coordinates (row, col).
        Grid IDs are numbered 0 to 1599 for a 40x40 grid.
        
        Args:
            grid_id: Grid ID (0 to grid_size^2 - 1)
        
        Returns:
            Tuple of (row, col) coordinates
        """
        row = grid_id // self.grid_size
        col = grid_id % self.grid_size
        return row, col
    
    def _coordinates_to_grid_id(self, row: int, col: int) -> int:
        """
        Convert grid coordinates to grid ID.
        
        Args:
            row: Row index (0 to grid_size-1)
            col: Column index (0 to grid_size-1)
        
        Returns:
            Grid ID
        """
        return row * self.grid_size + col
    
    def _calculate_distance(self, grid_id_1: int, grid_id_2: int) -> float:
        """
        Calculate Euclidean distance between two grids.
        
        Args:
            grid_id_1: First grid ID
            grid_id_2: Second grid ID
        
        Returns:
            Euclidean distance in grid units
        """
        row1, col1 = self._grid_id_to_coordinates(grid_id_1)
        row2, col2 = self._grid_id_to_coordinates(grid_id_2)
        
        distance = np.sqrt((row1 - row2) ** 2 + (col1 - col2) ** 2)
        
        # Avoid division by zero (for current location itself)
        if distance < 0.1:
            distance = 0.1
        
        return distance
    
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
                distance = self._calculate_distance(current_grid_id, grid_id)
                if distance <= self.radius:
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
        return_scores: bool = False
    ) -> Dict[str, List]:
        """
        Get candidate locations for each POI category using the gravity model.
        
        For each POI category, calculates gravity scores for all grids within radius,
        then returns the top-n grids with highest scores.
        
        Args:
            current_grid_id: Current location grid ID
            return_scores: If True, return tuples of (grid_id, score); 
                          If False, return only grid_ids
        
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
            
            # Sort by score (descending) and select top-n
            scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scores[:self.top_n]
            
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
        current_row, current_col = self._grid_id_to_coordinates(current_grid_id)
        
        print(f"\n{'='*70}")
        print(f"Candidate Locations for Current Grid: {current_grid_id}")
        print(f"Current Position: (row={current_row}, col={current_col})")
        print(f"Search Radius: {self.radius} grids")
        print(f"{'='*70}")
        
        candidates = self.get_candidate_locations(current_grid_id, return_scores=True)
        
        for category, top_grids in candidates.items():
            print(f"\n{category}:")
            for i, (grid_id, score) in enumerate(top_grids, 1):
                row, col = self._grid_id_to_coordinates(grid_id)
                distance = self._calculate_distance(current_grid_id, grid_id)
                print(f"  {i}. Grid {grid_id} (row={row}, col={col}) - "
                      f"Score: {score:.4f}, Distance: {distance:.2f}")
        
        print(f"\n{'='*70}")
        
        # Print top-k overall candidates
        print(f"\nTop-{self.top_n} Overall Candidates (by average score):")
        top_k_candidates = self.get_top_k_candidates(current_grid_id, k=self.top_n)
        for i, (grid_id, avg_score) in enumerate(top_k_candidates, 1):
            row, col = self._grid_id_to_coordinates(grid_id)
            distance = self._calculate_distance(current_grid_id, grid_id)
            print(f"  {i}. Grid {grid_id} (row={row}, col={col}) - "
                  f"Avg Score: {avg_score:.4f}, Distance: {distance:.2f}")
        
        print(f"{'='*70}\n")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the gravity model configuration.
        
        Returns:
            Dictionary with configuration statistics
        """
        return {
            'city': self.city,
            'weight': self.weight,
            'top_n': self.top_n,
            'radius': self.radius,
            'grid_size': self.grid_size,
            'total_grids': self.grid_size ** 2,
            'poi_categories': len(self.poi_categories),
            'poi_locations': len(self.poi_data)
        }
    
    def print_statistics(self):
        """Print gravity model statistics."""
        stats = self.get_statistics()
        
        print(f"\n{'='*50}")
        print("Gravity Model Statistics")
        print(f"{'='*50}")
        print(f"City: {stats['city']}")
        print(f"Weight Parameter: {stats['weight']}")
        print(f"Top-N Candidates per Category: {stats['top_n']}")
        print(f"Search Radius: {stats['radius']} grids")
        print(f"Grid System: {stats['grid_size']}x{stats['grid_size']} = {stats['total_grids']} grids")
        print(f"POI Categories: {stats['poi_categories']}")
        print(f"POI Locations: {stats['poi_locations']}")
        print(f"{'='*50}\n")
