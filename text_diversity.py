"""
Quick diversity test for MobilityPredictor.
Creates a synthetic trajectory with multiple distinct locations
to see if the LLM produces diverse next-location predictions.
"""

import json
from datetime import datetime, timedelta

from model.Predictor import MobilityPredictor


def make_synthetic_trajectory() -> list[dict]:
    """
    Build a synthetic observation trajectory with richer movement:
    - Starts residential, moves to office, then mall, then park, then transit hub.
    """
    base_time = datetime(2024, 6, 1, 8, 0, 0)
    grid_seq = [
        321, 321,         # residential stay
        845, 845,         # office
        1290,             # mall
        733,              # park
        1102              # transit hub
    ]
    traj = []
    for i, gid in enumerate(grid_seq):
        traj.append({
            "location_id": gid,
            "timestamp": (base_time + timedelta(minutes=10 * i)).isoformat()
        })
    return traj


def main():
    # Instantiate predictor with your usual settings
    predictor = MobilityPredictor(
        llm_model_name="Gemma-2-2B",
        llm_model_path="/datadisk",
        rag_database_path="/workspace/China_Journal/util/rag_database",
        city="beijing",
        top_k_predictions=10,
        rag_top_m_samples=5,
        gravity_top_n_candidates=5,
        gravity_weight=1.0,
        gravity_radius=10,
        use_quantization=True,
        use_lora=False,
        verbose=True
    )

    observation = make_synthetic_trajectory()

    # Use beam search with sampling to encourage diversity
    predictions, _, candidate_list = predictor.predict(
        observation_trajectory=observation,
        ground_truth=None,
        print_prompt=True,
        use_beam_search=True
    )

    print("\n=== Synthetic Trajectory ===")
    print(json.dumps(observation, indent=2, ensure_ascii=False))

    print("\n=== Predictions (grid_id, confidence) ===")
    for rank, (gid, conf) in enumerate(predictions, 1):
        print(f"{rank:2d}. Grid {gid} — conf={conf:.4f}")

    print("\nTotal candidate grids considered:", len(candidate_list))


if __name__ == "__main__":
    main()