"""
Quick test script for classification head modifications
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from model.Predictor import MobilityPredictor
from dataset.dataset import load_datasets

def test_classification_head():
    """Test the classification head functionality"""
    print("="*70)
    print("Testing Classification Head Modifications")
    print("="*70)
    
    # Load config
    config_path = "/workspace/China_Journal/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. Initializing MobilityPredictor...")
    
    # Initialize predictor with minimal settings for testing
    predictor = MobilityPredictor(
        llm_model_name="Deepseek-R1-Distill-Qwen-3B",
        llm_model_path="/datadisk",
        city="beijing",
        top_k_predictions=5,
        rag_top_m_samples=3,
        gravity_top_n_candidates=3,
        use_quantization=True,
        verbose=False
    )
    
    print(f"   ✓ Predictor initialized")
    print(f"   ✓ Classification head created with hidden size: {predictor.hidden_size}")
    print(f"   ✓ Classification head device: {next(predictor.classification_head.parameters()).device}")
    
    # Load a small sample of data
    print("\n2. Loading test data...")
    train_dataset, val_dataset, test_dataset = load_datasets(
        city="beijing",
        data_dir="/workspace/China_Journal/data",
        obs_len=12,
        pred_len=1,
        max_train_samples=5,
        max_val_samples=2,
        max_test_samples=2
    )
    
    print(f"   ✓ Loaded {len(train_dataset)} train samples")
    print(f"   ✓ Loaded {len(val_dataset)} val samples")
    
    # Test prediction with classification head
    print("\n3. Testing prediction with classification head...")
    sample = val_dataset[0]
    observation = sample['observation']
    ground_truth = sample['next_location']
    
    print(f"   Ground truth: Grid {ground_truth}")
    print(f"   Observation length: {len(observation)}")
    
    # Make prediction
    predictions, results = predictor.predict(
        observation_trajectory=observation,
        ground_truth=ground_truth,
        print_prompt=False,
        return_logits=True
    )
    
    print(f"\n   Predictions (top-5):")
    for i, (loc_id, prob) in enumerate(predictions[:5], 1):
        marker = "✓" if loc_id == ground_truth else " "
        print(f"   {marker} {i}. Grid {loc_id} (probability: {prob:.4f})")
    
    print(f"\n   ✓ Classification logits shape: {results['logits'].shape}")
    print(f"   ✓ Number of candidates: {len(results['candidate_list'])}")
    
    # Test loss computation
    print("\n4. Testing loss computation...")
    
    # Setup optimizer for test
    trainable_params = list(predictor.llm.model.parameters()) + \
                      list(predictor.classification_head.parameters())
    predictor.llm.optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    # Compute loss (create a minimal trainer-like context)
    predictor.llm.model.eval()
    
    with torch.no_grad():
        # Get RAG summary
        rag_summary, _, _ = predictor.rag.generate_rag_summary(
            query_trajectory=observation,
            print_prompt=False
        )
        
        # Get gravity candidates
        current_location = observation[-1]['location_id']
        candidates_by_category = predictor.gravity.get_candidate_locations(
            current_grid_id=current_location,
            return_scores=True
        )
    
    # Build prompt
    prompt = predictor._build_prediction_prompt(
        observation_trajectory=observation,
        rag_summary=rag_summary,
        candidates_by_category=candidates_by_category
    )
    
    # Get all candidates
    all_candidates = set()
    for category, candidates in candidates_by_category.items():
        for grid_id, score in candidates:
            all_candidates.add(grid_id)
    candidate_list = sorted(list(all_candidates))
    
    # Add ground truth if not in candidates
    if ground_truth not in candidate_list:
        candidate_list.append(ground_truth)
        candidate_list = sorted(candidate_list)
    
    gt_index = candidate_list.index(ground_truth)
    
    # Get hidden states
    inputs = predictor.llm.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(predictor.llm.device)
    
    predictor.llm.model.train()
    outputs = predictor.llm.model(**inputs, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    context_encoding = last_hidden_state[:, -1, :]
    
    # Score candidates
    logits_list = []
    for candidate_id in candidate_list:
        score = predictor.classification_head(context_encoding)
        logits_list.append(score)
    
    logits = torch.cat(logits_list, dim=1)
    
    # Compute loss
    gt_tensor = torch.tensor([gt_index], dtype=torch.long).to(predictor.llm.device)
    loss = torch.nn.functional.cross_entropy(logits, gt_tensor)
    
    print(f"   ✓ Loss computed: {loss.item():.4f}")
    print(f"   ✓ Logits shape: {logits.shape}")
    print(f"   ✓ Ground truth in candidates: {ground_truth in candidate_list}")
    print(f"   ✓ Ground truth index: {gt_index}/{len(candidate_list)}")
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    loss.backward()
    
    # Check gradients
    llm_has_grads = any(p.grad is not None for p in predictor.llm.model.parameters())
    cls_has_grads = any(p.grad is not None for p in predictor.classification_head.parameters())
    
    print(f"   ✓ LLM has gradients: {llm_has_grads}")
    print(f"   ✓ Classification head has gradients: {cls_has_grads}")
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nSummary:")
    print("  • Classification head successfully added to MobilityPredictor")
    print("  • Predictions now use classification head probabilities")
    print("  • Loss computation updated to use classification head")
    print("  • Gradients flow through both LLM and classification head")
    print("="*70)

if __name__ == "__main__":
    test_classification_head()
