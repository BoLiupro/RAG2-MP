"""
Test script for all-grids prediction with gradient monitoring
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from model.Predictor import MobilityPredictor
from dataset.dataset import load_datasets

def test_all_grids_prediction():
    """Test prediction over all 1600 grids with gradient checks"""
    print("="*70)
    print("Testing All-Grids Prediction with Gradient Monitoring")
    print("="*70)
    
    # Load config
    config_path = "/workspace/China_Journal/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. Initializing MobilityPredictor...")
    
    # Initialize predictor
    predictor = MobilityPredictor(
        llm_model_name="Deepseek-R1-Distill-Qwen-3B",
        llm_model_path="/datadisk",
        city="beijing",
        top_k_predictions=10,
        rag_top_m_samples=3,
        gravity_top_n_candidates=3,
        use_quantization=True,
        verbose=False
    )
    
    print(f"   ✓ Predictor initialized")
    print(f"   ✓ Classification head output size: {predictor.num_grids}")
    print(f"   ✓ Grid size: {predictor.grid_size}x{predictor.grid_size}")
    
    # Freeze LLM, train classification head
    for p in predictor.llm.model.parameters():
        p.requires_grad = False
    for p in predictor.classification_head.parameters():
        p.requires_grad = True
    
    print(f"   ✓ LLM frozen, classification head trainable")
    
    # Load data
    print("\n2. Loading test data...")
    train_dataset, val_dataset, test_dataset = load_datasets(
        city="beijing",
        data_dir="/workspace/China_Journal/data",
        obs_len=12,
        pred_len=1,
        max_train_samples=3,
        max_val_samples=2,
        max_test_samples=2
    )
    
    print(f"   ✓ Loaded {len(train_dataset)} train samples")
    
    # Setup optimizer
    trainable_params = list(predictor.classification_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
    
    print("\n3. Testing forward pass...")
    sample = train_dataset[0]
    observation = sample['observation']
    ground_truth = sample['next_location']
    
    print(f"   Ground truth: Grid {ground_truth}")
    
    # Get RAG summary and gravity candidates
    predictor.llm.model.eval()
    predictor.classification_head.eval()
    with torch.no_grad():
        rag_summary, _, _ = predictor.rag.generate_rag_summary(
            query_trajectory=observation,
            print_prompt=False
        )
        
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
    
    # Tokenize
    inputs = predictor.llm.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(predictor.llm.device)
    
    # Get hidden states
    with torch.no_grad():
        outputs = predictor.llm.model(**inputs, output_hidden_states=True)
    
    last_hidden_state = outputs.hidden_states[-1]
    context_encoding = last_hidden_state[:, -1, :].detach()
    
    # Convert to float32 for classification head
    if predictor.use_fp32_head:
        context_encoding = context_encoding.float()
    
    print(f"   ✓ Context encoding shape: {context_encoding.shape}")
    print(f"   ✓ Context encoding dtype: {context_encoding.dtype}")
    
    # Forward through classification head
    predictor.classification_head.train()
    logits = predictor.classification_head(context_encoding)
    
    print(f"   ✓ Logits shape: {logits.shape}")
    print(f"   ✓ Predicting over {logits.shape[1]} grids")
    print(f"   ✓ Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print(f"   ✓ Logits mean: {logits.mean().item():.2f}")
    
    # Compute loss
    gt_tensor = torch.tensor([ground_truth], dtype=torch.long).to(predictor.llm.device)
    loss = torch.nn.functional.cross_entropy(logits, gt_tensor)
    
    print(f"   ✓ Loss: {loss.item():.4f}")
    
    # Check for NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"   ✗ ERROR: Loss is NaN or Inf!")
        return False
    
    print("\n4. Testing backward pass and gradient flow...")
    optimizer.zero_grad()
    loss.backward()
    
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(predictor.classification_head.parameters(), max_norm=0.5)
    
    # Check gradients
    total_norm = 0.0
    has_nan = False
    has_inf = False
    grad_stats = []
    
    for name, p in predictor.classification_head.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            grad_stats.append((name, param_norm, p.grad.min().item(), p.grad.max().item()))
            
            if torch.isnan(p.grad).any():
                has_nan = True
            if torch.isinf(p.grad).any():
                has_inf = True
    
    total_norm = total_norm ** 0.5
    
    print(f"   ✓ Total gradient norm: {total_norm:.4f}")
    
    if has_nan:
        print(f"   ✗ ERROR: NaN gradients detected!")
        return False
    if has_inf:
        print(f"   ✗ ERROR: Inf gradients detected!")
        return False
    
    print(f"   ✓ No NaN/Inf in gradients")
    
    print(f"\n   Gradient statistics per layer:")
    for name, norm, min_val, max_val in grad_stats:
        print(f"     {name}: norm={norm:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
    
    # Optimizer step
    optimizer.step()
    print(f"   ✓ Optimizer step completed")
    
    print("\n5. Testing second sample (check for stability)...")
    sample2 = train_dataset[1] if len(train_dataset) > 1 else train_dataset[0]
    observation2 = sample2['observation']
    ground_truth2 = sample2['next_location']
    
    # Get RAG summary again
    with torch.no_grad():
        rag_summary2, _, _ = predictor.rag.generate_rag_summary(
            query_trajectory=observation2,
            print_prompt=False
        )
    
    # Check if RAG still works
    if "!" * 5 in rag_summary2:
        print(f"   ✗ ERROR: RAG output corrupted (!!!!)!")
        return False
    
    print(f"   ✓ RAG summary still normal")
    print(f"   ✓ Ground truth: Grid {ground_truth2}")
    
    # Forward pass
    current_location2 = observation2[-1]['location_id']
    with torch.no_grad():
        candidates2 = predictor.gravity.get_candidate_locations(
            current_grid_id=current_location2,
            return_scores=True
        )
    
    prompt2 = predictor._build_prediction_prompt(
        observation_trajectory=observation2,
        rag_summary=rag_summary2,
        candidates_by_category=candidates2
    )
    
    inputs2 = predictor.llm.tokenizer(
        prompt2,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(predictor.llm.device)
    
    with torch.no_grad():
        outputs2 = predictor.llm.model(**inputs2, output_hidden_states=True)
    
    context2 = outputs2.hidden_states[-1][:, -1, :].detach()
    
    # Convert to float32 for classification head
    if predictor.use_fp32_head:
        context2 = context2.float()
    
    # Debug: check context encoding
    print(f"   Context encoding stats:")
    print(f"     Shape: {context2.shape}")
    print(f"     Range: [{context2.min().item():.4f}, {context2.max().item():.4f}]")
    print(f"     Mean: {context2.mean().item():.4f}")
    print(f"     Has NaN: {torch.isnan(context2).any().item()}")
    print(f"     Has Inf: {torch.isinf(context2).any().item()}")
    
    logits2 = predictor.classification_head(context2)
    
    # Debug: check logits
    print(f"   Logits stats:")
    print(f"     Shape: {logits2.shape}")
    print(f"     Range: [{logits2.min().item():.4f}, {logits2.max().item():.4f}]")
    print(f"     Mean: {logits2.mean().item():.4f}")
    print(f"     Has NaN: {torch.isnan(logits2).any().item()}")
    print(f"     Has Inf: {torch.isinf(logits2).any().item()}")
    
    gt_tensor2 = torch.tensor([ground_truth2], dtype=torch.long).to(predictor.llm.device)
    loss2 = torch.nn.functional.cross_entropy(logits2, gt_tensor2)
    
    print(f"   ✓ Loss: {loss2.item():.4f}")
    
    if torch.isnan(loss2) or torch.isinf(loss2):
        print(f"   ✗ ERROR: Loss is NaN or Inf on second sample!")
        return False
    
    # Backward
    optimizer.zero_grad()
    loss2.backward()
    
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(predictor.classification_head.parameters(), max_norm=0.5)
    
    total_norm2 = sum(p.grad.data.norm(2).item() ** 2 for p in predictor.classification_head.parameters() if p.grad is not None) ** 0.5
    print(f"   ✓ Gradient norm: {total_norm2:.4f}")
    
    optimizer.step()
    print(f"   ✓ Second sample training successful")
    
    print("\n6. Testing prediction method...")
    predictions, results = predictor.predict(
        observation_trajectory=observation,
        ground_truth=ground_truth,
        print_prompt=False,
        return_logits=True
    )
    
    print(f"   ✓ Predictions returned: {len(predictions)} locations")
    print(f"   ✓ Top-5 predictions:")
    for i, (loc_id, prob) in enumerate(predictions[:5], 1):
        marker = "✓" if loc_id == ground_truth else " "
        print(f"     {marker} {i}. Grid {loc_id} (prob: {prob:.4f})")
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nSummary:")
    print("  • Classification head predicts over all 1600 grids")
    print("  • Gradients flow correctly (no NaN/Inf)")
    print("  • LLM remains frozen and stable")
    print("  • Training works across multiple samples")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = test_all_grids_prediction()
    if not success:
        print("\n✗ Tests FAILED")
        exit(1)
