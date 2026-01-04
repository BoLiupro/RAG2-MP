"""
Test script to verify RAG JSON parsing improvements.
Tests the handling of reasoning content from Deepseek-R1 model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.RAG import MobilityRAG
import json

def test_parse_json_response():
    """Test the improved JSON parsing with various response formats."""
    
    rag = MobilityRAG(llm=None)  # We only need the parsing method
    
    # Test case 1: Clean JSON response
    print("=" * 80)
    print("Test 1: Clean JSON response")
    print("=" * 80)
    clean_json = """{
  "next_locations": [
    {
      "avg_distance_km": 2.5,
      "area_type": "residential",
      "evidence_basis": "time similarity, POI transition"
    }
  ],
  "spatial_patterns": {
    "distance_range_km": "1.0–5.0 km",
    "movement_type": "short-range"
  },
  "temporal_patterns": {
    "dominant_time_windows": "morning_peak",
    "temporal_consistency": "high"
  },
  "pattern_confidence": "high"
}"""
    
    try:
        result = rag._parse_json_response(clean_json)
        print("✅ SUCCESS: Parsed clean JSON")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test case 2: JSON with thinking tags (Deepseek-R1 style)
    print("\n" + "=" * 80)
    print("Test 2: JSON with <think> tags (Deepseek-R1)")
    print("=" * 80)
    json_with_thinking = """<think>
Let me analyze the patterns...
The user is at location 1234, and from the retrieved samples, I see that:
- 3 samples went to location 5678
- Distance is around 2-3 km
- Mostly in morning hours
This suggests a commute pattern to work area.
</think>

{
  "next_locations": [
    {
      "avg_distance_km": 2.5,
      "area_type": "commercial",
      "evidence_basis": "3 historical visits, morning commute pattern"
    }
  ],
  "spatial_patterns": {
    "distance_range_km": "2.0–3.0 km",
    "movement_type": "short-range"
  },
  "temporal_patterns": {
    "dominant_time_windows": "morning_peak",
    "temporal_consistency": "high"
  },
  "pattern_confidence": "high"
}"""
    
    try:
        result = rag._parse_json_response(json_with_thinking)
        print("✅ SUCCESS: Parsed JSON with <think> tags")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test case 3: JSON with markdown code blocks
    print("\n" + "=" * 80)
    print("Test 3: JSON with markdown code blocks")
    print("=" * 80)
    json_with_markdown = """Here is the analysis:

```json
{
  "next_locations": [
    {
      "avg_distance_km": 1.2,
      "area_type": "residential",
      "evidence_basis": "close proximity, residential area"
    }
  ],
  "spatial_patterns": {
    "distance_range_km": "0.5–2.0 km",
    "movement_type": "short-range"
  },
  "temporal_patterns": {
    "dominant_time_windows": "evening",
    "temporal_consistency": "medium"
  },
  "pattern_confidence": "medium"
}
```"""
    
    try:
        result = rag._parse_json_response(json_with_markdown)
        print("✅ SUCCESS: Parsed JSON with markdown")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Test case 6: JSON with </think> but no <think> (truncated or prompt echo)
    print("\n" + "=" * 80)
    print("Test 6: JSON with </think> but no <think>")
    print("=" * 80)
    json_missing_start_tag = """Answer directly...
...
}
</think>

```json
{
  "next_locations": [
    {
      "avg_distance_km": 2.5,
      "area_type": "residential",
      "evidence_basis": "time similarity"
    }
  ],
  "spatial_patterns": {
    "distance_range_km": "1.0–5.0 km",
    "movement_type": "short-range"
  },
  "temporal_patterns": {
    "dominant_time_windows": "morning_peak",
    "temporal_consistency": "high"
  },
  "pattern_confidence": "high"
}
```"""
    
    try:
        result = rag._parse_json_response(json_missing_start_tag)
        print("✅ SUCCESS: Parsed JSON with missing <think> tag")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test case 4: JSON with both thinking and extra text
    print("\n" + "=" * 80)
    print("Test 4: JSON with thinking tags and extra explanatory text")
    print("=" * 80)
    json_complex = """<think>
Analyzing mobility patterns...
Retrieved 5 similar samples
Most common next location: 7890
</think>

Based on the analysis, here's the structured summary:

{
  "next_locations": [
    {
      "avg_distance_km": 5.5,
      "area_type": "commercial, shopping",
      "evidence_basis": "5 historical visits, distance range, POI type"
    }
  ],
  "spatial_patterns": {
    "distance_range_km": "4.0–7.0 km",
    "movement_type": "medium-range"
  },
  "temporal_patterns": {
    "dominant_time_windows": "afternoon",
    "temporal_consistency": "medium"
  },
  "pattern_confidence": "high"
}

This indicates a shopping trip pattern."""
    
    try:
        result = rag._parse_json_response(json_complex)
        print("✅ SUCCESS: Parsed complex JSON")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test case 5: Invalid JSON (should fail gracefully)
    print("\n" + "=" * 80)
    print("Test 5: Invalid JSON (should fail with informative error)")
    print("=" * 80)
    invalid_json = """This is just text without any JSON structure."""
    
    try:
        result = rag._parse_json_response(invalid_json)
        print("❌ UNEXPECTED: Should have failed but didn't")
    except ValueError as e:
        print(f"✅ EXPECTED FAILURE: {e}")
    except Exception as e:
        print(f"⚠️  UNEXPECTED ERROR TYPE: {e}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_parse_json_response()
