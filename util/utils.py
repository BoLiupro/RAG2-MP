"""
Utility Functions for Mobility Prediction
Common functions used across multiple modules.
"""

import numpy as np
from typing import Tuple, List
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2


# City grid boundaries for lat/lon coordinate mapping
# Each city has a 40x40 grid system
CITY_BOUNDARIES = {
    'beijing': {
        'lat_min': 39.84,
        'lat_max': 40.00,
        'lon_min': 116.31,
        'lon_max': 116.47,
        'grid_size': 40
    },
    'shenzhen': {
        'lat_min': 22.51,
        'lat_max': 22.67,
        'lon_min': 114.02,
        'lon_max': 114.18,
        'grid_size': 40
    },
    'nanchang': {
        'lat_min': 28.58,
        'lat_max': 28.74,
        'lon_min': 115.78,
        'lon_max': 115.94,
        'grid_size': 40
    }
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
    
    Returns:
        Distance in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    
    return distance


def grid_id_to_latlon(grid_id: int, city: str = 'beijing', grid_size: int = 40) -> Tuple[float, float]:
    """
    Convert grid ID to latitude and longitude coordinates (center of the grid).
    
    Args:
        grid_id: Grid ID (0 to grid_size^2 - 1)
        city: City name ('beijing', 'shenzhen', 'nanchang')
        grid_size: Size of the grid system (default 40x40)
    
    Returns:
        Tuple of (latitude, longitude) for the center of the grid
    """
    city_lower = city.lower()
    if city_lower not in CITY_BOUNDARIES:
        raise ValueError(f"Unknown city: {city}. Must be one of {list(CITY_BOUNDARIES.keys())}")
    
    bounds = CITY_BOUNDARIES[city_lower]
    
    # Get row and column from grid_id
    row = grid_id // grid_size
    col = grid_id % grid_size
    
    # Calculate lat/lon for the center of the grid cell
    lat_step = (bounds['lat_max'] - bounds['lat_min']) / grid_size
    lon_step = (bounds['lon_max'] - bounds['lon_min']) / grid_size
    
    lat = bounds['lat_min'] + (row + 0.5) * lat_step
    lon = bounds['lon_min'] + (col + 0.5) * lon_step
    
    return lat, lon


def calculate_grid_distance(grid_id_1: int, grid_id_2: int, city: str = 'beijing', grid_size: int = 40) -> float:
    """
    Calculate distance between two grids using Haversine formula on lat/lon coordinates.
    
    Args:
        grid_id_1: First grid ID
        grid_id_2: Second grid ID
        city: City name for coordinate conversion
        grid_size: Size of the grid system (default 40x40)
    
    Returns:
        Distance in kilometers
    """
    # Get lat/lon coordinates for both grids
    lat1, lon1 = grid_id_to_latlon(grid_id_1, city, grid_size)
    lat2, lon2 = grid_id_to_latlon(grid_id_2, city, grid_size)
    
    # Calculate haversine distance
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Avoid division by zero (for current location itself)
    if distance < 0.01:  # Less than 10 meters
        distance = 0.001
    
    return distance


def grid_id_to_coordinates(grid_id: int, grid_size: int = 40) -> Tuple[int, int]:
    """
    Convert grid ID to grid coordinates (row, col).
    
    Args:
        grid_id: Grid ID (0 to grid_size^2 - 1)
        grid_size: Size of the grid system (default 40x40)
    
    Returns:
        Tuple of (row, col) coordinates
    """
    row = grid_id // grid_size
    col = grid_id % grid_size
    return row, col


def coordinates_to_grid_id(row: int, col: int, grid_size: int = 40) -> int:
    """
    Convert grid coordinates to grid ID.
    
    Args:
        row: Row index (0 to grid_size-1)
        col: Column index (0 to grid_size-1)
        grid_size: Size of the grid system (default 40x40)
    
    Returns:
        Grid ID
    """
    return row * grid_size + col


def calculate_trajectory_distances(trajectory: List[int], city: str = 'beijing', grid_size: int = 40) -> List[float]:
    """
    Calculate distances between consecutive locations in a trajectory.
    Returns distances FROM previous location (i.e., distance from location i-1 to location i).
    
    Args:
        trajectory: List of grid IDs representing a trajectory
        city: City name for coordinate conversion
        grid_size: Size of the grid system (default 40x40)
    
    Returns:
        List of distances (length = len(trajectory), first element is 0.0 for starting point)
    """
    if len(trajectory) == 0:
        return []
    
    # First location has no previous distance
    distances = [0.0]
    
    for i in range(1, len(trajectory)):
        dist = calculate_grid_distance(trajectory[i-1], trajectory[i], city, grid_size)
        distances.append(dist)
    
    return distances


def format_trajectory_with_distances(
    trajectory: List[dict],
    city: str = 'beijing',
    grid_size: int = 40,
    include_poi: bool = False,
    poi_data: dict = None
) -> str:
    """
    Format trajectory with distance information for prompts.
    Shows distance FROM previous location.
    
    Args:
        trajectory: List of trajectory points with 'location_id' and 'timestamp'
        city: City name for coordinate conversion
        grid_size: Size of the grid system
        include_poi: Whether to include POI information
        poi_data: Dictionary mapping location_id to POI features
    
    Returns:
        Formatted trajectory string with distances from previous location
    """
    if not trajectory:
        return "Empty trajectory"
    
    formatted = []
    locations = [p['location_id'] for p in trajectory]
    distances = calculate_trajectory_distances(locations, city, grid_size)
    
    for i, point in enumerate(trajectory):
        loc_id = point.get('location_id', 'unknown')
        timestamp = point.get('timestamp', '')
        
        # Parse timestamp
        try:
            if isinstance(timestamp, str):
                dt = datetime.strptime(timestamp, '%Y%m%d %H:%M')
                time_str = dt.strftime('%Y-%m-%d %H:%M')
                weekday = dt.strftime('%A')
            else:
                time_str = str(timestamp)
                weekday = "Unknown"
        except:
            time_str = str(timestamp)
            weekday = "Unknown"
        
        # Build line with distance info
        line = f"{i+1}. Location {loc_id}, {time_str} ({weekday})"
        
        # Add distance from previous location
        if i == 0:
            line += " (starting point)"
        else:
            line += f" (distance from previous: {distances[i]:.2f} km)"
        
        # Add POI information if available
        if include_poi and poi_data and loc_id in poi_data:
            poi_features = poi_data[loc_id]
            # Find dominant POI types
            poi_types = []
            for poi_type, percentage in poi_features.items():
                if percentage > 10:  # Only show significant POI types
                    poi_types.append(f"{poi_type}: {percentage:.1f}%")
            if poi_types:
                line += f" [POI: {', '.join(poi_types[:2])}]"  # Top 2
        
        formatted.append(line)
    
    return "\n".join(formatted)


def get_mobility_mode(city: str) -> str:
    """
    Get the mobility mode description for a city.
    
    Args:
        city: City name ('shenzhen', 'beijing', 'nanchang')
    
    Returns:
        Mobility mode description
    """
    return "general mobility (may include walking, public transport, taxi, etc.)"


def parse_grid_ids_from_text(text: str) -> List[int]:
    """
    Parse grid IDs from text using various patterns.
    
    Args:
        text: Text containing grid ID references
    
    Returns:
        List of parsed grid IDs
    """
    import re
    
    # Try various patterns
    patterns = [
        r'Grid\s+(\d+)',
        r'Location\s+(\d+)',
        r'grid\s+(\d+)',
        r'location\s+(\d+)',
        r'\b(\d+)\b'  # Any standalone number
    ]
    
    grid_ids = []
    seen = set()
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            grid_id = int(match)
            if grid_id not in seen and 0 <= grid_id < 1600:  # Valid grid range
                grid_ids.append(grid_id)
                seen.add(grid_id)
    
    return grid_ids


def calculate_accuracy_at_k(predictions: List[List[int]], ground_truths: List[int], k: int) -> float:
    """
    Calculate accuracy@k metric.
    
    Args:
        predictions: List of prediction lists (each list contains predicted grid IDs)
        ground_truths: List of ground truth grid IDs
        k: Top-k to consider
    
    Returns:
        Accuracy@k score
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    hits = 0
    for pred_list, gt in zip(predictions, ground_truths):
        if gt in pred_list[:k]:
            hits += 1
    
    return hits / len(ground_truths)


def calculate_mrr(predictions: List[List[int]], ground_truths: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        predictions: List of prediction lists (each list contains predicted grid IDs)
        ground_truths: List of ground truth grid IDs
    
    Returns:
        MRR score
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    rr_sum = 0.0
    for pred_list, gt in zip(predictions, ground_truths):
        if gt in pred_list:
            rank = pred_list.index(gt) + 1
            rr_sum += 1.0 / rank
    
    return rr_sum / len(ground_truths)


def format_candidates_with_distances(
    candidates: List[Tuple[int, float]],
    current_location: int,
    city: str = 'beijing',
    grid_size: int = 40,
    include_scores: bool = True
) -> str:
    """
    Format candidate locations with distance information.
    
    Args:
        candidates: List of (grid_id, score) tuples
        current_location: Current grid ID
        city: City name for coordinate conversion
        grid_size: Size of the grid system
        include_scores: Whether to include gravity scores
    
    Returns:
        Formatted string with candidates and distances in kilometers
    """
    formatted = []
    for grid_id, score in candidates:
        dist = calculate_grid_distance(current_location, grid_id, city, grid_size)
        if include_scores:
            formatted.append(f"Grid {grid_id} (distance: {dist:.2f} km, score: {score:.2f})")
        else:
            formatted.append(f"Grid {grid_id} (distance: {dist:.2f} km)")
    
    return ", ".join(formatted)
