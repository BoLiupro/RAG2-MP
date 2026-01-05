"""
Test script to verify the simplified prompt formatting in MobilityPredictor.
"""

import sys
sys.path.append('/workspace/China_Journal')

from model.Predictor import MobilityPredictor
from datetime import datetime

# Create a simple test trajectory
test_trajectory = [
    {'location_id': 100, 'timestamp': '20220520 08:00'},
    {'location_id': 150, 'timestamp': '20220520 08:30'},
    {'location_id': 200, 'timestamp': '20220520 09:00'},
    {'location_id': 250, 'timestamp': '20220520 09:30'},
]

# Mock RAG summary
mock_rag_summary = """
{
  "next_locations": [
    {
      "grid_id": 300,
      "frequency": 5,
      "avg_distance_km": 2.3,
      "area_type": "Shopping & Consumer Goods",
      "evidence_basis": "morning commute pattern"
    }
  ],
  "spatial_patterns": {
    "distance_range_km": "1.5-3.0",
    "movement_type": "medium-range"
  },
  "temporal_patterns": {
    "dominant_time_windows": "morning_peak",
    "temporal_consistency": "high"
  },
  "pattern_confidence": "high"
}
"""

# Mock candidates by category
mock_candidates = {
    'Dining & Cusine': [(260, 0.8), (270, 0.7), (280, 0.6)],
    'Shopping & Consumer Goods': [(300, 0.9), (310, 0.75), (320, 0.65)],
    'Residential': [(250, 1.0), (255, 0.85), (265, 0.7)],
    'Transportation Facilities': [(290, 0.8), (295, 0.7), (305, 0.6)]
}

print("=" * 80)
print("Testing Simplified Prompt Format")
print("=" * 80)

try:
    # Initialize predictor (this may fail if model not available, but we can still test the prompt)
    print("\n1. Testing trajectory compact formatting...")
    print("-" * 80)
    
    # We'll create a minimal mock to test just the formatting functions
    class MockLLM:
        def __init__(self):
            self.model_name = "test-model"
    
    class MockRAG:
        def __init__(self):
            self.poi_data = {
                100: {'Residential_count': 50, 'Shopping & Consumer Goods_count': 20},
                150: {'Transportation Facilities_count': 60, 'Residential_count': 15},
                200: {'Shopping & Consumer Goods_count': 70, 'Dining & Cusine_count': 20},
                250: {'Dining & Cusine_count': 80, 'Life Services_count': 10}
            }
    
    # Create a mock predictor instance
    predictor = MobilityPredictor.__new__(MobilityPredictor)
    predictor.city = 'beijing'
    predictor.rag = MockRAG()
    predictor.verbose = True
    
    # Test trajectory compact formatting
    compact_trajectory = predictor._format_trajectory_compact(test_trajectory)
    print("\nCompact Trajectory Format:")
    print(compact_trajectory)
    
    print("\n" + "=" * 80)
    print("2. Testing candidate compact formatting...")
    print("-" * 80)
    
    # Test candidates compact formatting
    current_location = test_trajectory[-1]['location_id']
    compact_candidates = predictor._format_candidates_compact(mock_candidates, current_location)
    print("\nCompact Candidates Format:")
    print(compact_candidates)
    
    print("\n" + "=" * 80)
    print("3. Testing full prompt building...")
    print("-" * 80)
    
    # Test full prompt
    full_prompt = predictor._build_final_prediction_prompt(
        test_trajectory,
        mock_rag_summary,
        mock_candidates
    )
    
    print("\nFull Prompt:")
    print(full_prompt)
    
    print("\n" + "=" * 80)
    print("✓ All formatting tests completed successfully!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
