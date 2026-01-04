"""
Test script to verify RAG summary integration in Predictor.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.Predictor import MobilityPredictor
from model.RAG import MobilityRAG
from model.LLM import MobilityLLM
import json

def test_predictor_prompt_with_json_summary():
    """Test that Predictor correctly includes JSON summary in prompt."""
    
    # Mock objects
    class MockLLM:
        def __init__(self):
            self.model_name = "mock"
    
    class MockGravity:
        pass
        
    # Initialize predictor with mocks
    # Note: Predictor initializes its own components, so we need to patch them or use valid args
    # For this simple test, we'll just instantiate it with minimal args and patch the method we want to test
    # But since we are testing _build_final_prediction_prompt which is a method of the instance,
    # we need a proper instance.
    
    # Let's create a subclass that overrides __init__ to accept mocks
    class TestablePredictor(MobilityPredictor):
        def __init__(self, config):
            self.city = config['data']['city']
            self.top_k_predictions = config['model']['top_k_predictions']
            self.llm = MockLLM()
            self.gravity = MockGravity()
            self.rag = None
            
    predictor = TestablePredictor(config)
    
    # Create a sample JSON summary string (as returned by modified RAG)
    json_summary = json.dumps({
        "next_locations": [
            {
                "avg_distance_from_previous_location_km": 2.5,
                "area_type_of_next_location": "residential",
                "reason": "time similarity"
            }
        ],
        "spatial_patterns": "short-range",
        "temporal_patterns": "morning_peak"
    }, indent=2)
    
    # Sample inputs
    observation = [
        {'location_id': 100, 'timestamp': '20220101 08:00'},
        {'location_id': 101, 'timestamp': '20220101 09:00'}
    ]
    
    candidates = {
        'residential_count': [(102, 0.8), (103, 0.6)],
        'commercial_count': [(200, 0.5)]
    }
    
    # Build prompt
    prompt = predictor._build_final_prediction_prompt(
        observation_trajectory=observation,
        rag_summary=json_summary,
        candidates_by_category=candidates
    )
    
    print("=" * 80)
    print("Generated Prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    
    # Verify JSON summary is present
    if "Similar Historical Patterns (JSON Summary):" in prompt:
        print("✅ SUCCESS: JSON Summary header found")
    else:
        print("❌ FAILED: JSON Summary header missing")
        
    if '"next_locations":' in prompt:
        print("✅ SUCCESS: JSON content found")
    else:
        print("❌ FAILED: JSON content missing")

if __name__ == "__main__":
    test_predictor_prompt_with_json_summary()
