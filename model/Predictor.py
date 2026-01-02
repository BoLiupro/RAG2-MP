"""
MobilityPredictor Module
This module integrates LLM+RAG and Gravity Model for mobility prediction.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from .LLM import MobilityLLM
from .RAG import MobilityRAG
from .Gravity import GravityModel


class MobilityPredictor:
    """
    Mobility Prediction System integrating LLM+RAG and Gravity Model.
    
    This class orchestrates the complete prediction pipeline:
    1. Retrieves similar trajectories using RAG
    2. Generates candidate locations using Gravity Model
    3. Combines both to make final predictions using LLM
    """
    
    def __init__(
        self,
        llm_model_name: str = "Deepseek-R1-Distill-Qwen-3B",
        llm_model_path: str = "/datadisk",
        rag_database_path: str = "/workspace/China_Journal/model/rag_database",
        poi_data_path: str = None,
        city: str = "beijing",
        top_k: int = 5,
        top_m: int = 5,
        top_n: int = 5,
        gravity_weight: float = 1.0,
        gravity_radius: int = 10,
        use_quantization: bool = True,
        use_lora: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the MobilityPredictor.
        
        Args:
            llm_model_name: Name of the LLM model
            llm_model_path: Path to the LLM model directory
            rag_database_path: Path to RAG database
            poi_data_path: Path to POI data CSV file
            city: City name
            top_k: Number of top predictions to return
            top_m: Number of similar samples to retrieve (RAG)
            top_n: Number of candidates per POI category (Gravity)
            gravity_weight: Weight parameter for gravity model
            gravity_radius: Search radius for gravity model (in grid units)
            use_quantization: Whether to use 4-bit quantization for LLM
            use_lora: Whether to use LoRA fine-tuning
            verbose: If True, print prompts and intermediate results
        """
        self.city = city
        self.top_k = top_k
        self.top_m = top_m
        self.top_n = top_n
        self.gravity_weight = gravity_weight
        self.gravity_radius = gravity_radius
        self.verbose = verbose
        
        print(f"\n{'='*70}")
        print("Initializing MobilityPredictor")
        print(f"{'='*70}")
        print(f"City: {city}")
        print(f"Top-K predictions: {top_k}")
        print(f"RAG top-M similar samples: {top_m}")
        print(f"Gravity top-N per category: {top_n}")
        print(f"Gravity weight: {gravity_weight}")
        print(f"Gravity radius: {gravity_radius}")
        print(f"Verbose mode: {verbose}")
        
        # Initialize LLM
        print(f"\nStep 1: Initializing LLM...")
        self.llm = MobilityLLM(
            model_name=llm_model_name,
            model_path=llm_model_path,
            use_quantization=use_quantization,
            use_lora=use_lora
        )
        
        # Initialize RAG module
        print(f"\nStep 2: Initializing RAG module...")
        self.rag = MobilityRAG(
            llm=self.llm,
            rag_database_path=rag_database_path,
            top_m=top_m,
            city=city,
            verbose=verbose
        )
        
        # Load RAG database
        if self.rag.load_rag_database():
            print("✓ RAG database loaded successfully")
        else:
            print("⚠ Warning: RAG database not loaded")
        
        # Set POI data path
        if poi_data_path is None:
            poi_data_path = f"/workspace/China_Journal/raw_data/{city}/{city}_grid_poi.csv"
        
        # Load POI data for RAG
        if self.rag.load_poi_data(poi_data_path):
            print("✓ POI data loaded for RAG")
        
        # Initialize Gravity Model
        print(f"\nStep 3: Initializing Gravity Model...")
        self.gravity = GravityModel(
            poi_data_path=poi_data_path,
            city=city,
            weight=gravity_weight,
            top_n=top_n,
            radius=gravity_radius
        )
        
        print(f"\n{'='*70}")
        print("MobilityPredictor initialized successfully!")
        print(f"{'='*70}\n")
    
    def predict(
        self,
        observation_trajectory: List[Dict[str, Any]],
        ground_truth: int = None,
        print_prompt: bool = None
    ) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
        """
        Make mobility prediction for the next location.
        
        Args:
            observation_trajectory: List of trajectory points, each with 'location_id' and 'timestamp'
            ground_truth: Ground truth next location (optional, for evaluation)
            print_prompt: If True, print the final prediction prompt (overrides self.verbose)
        
        Returns:
            Tuple of:
                - List of (location_id, confidence) tuples for top-K predictions
                - Dictionary with intermediate results (summary, candidates, etc.)
        """
        if print_prompt is None:
            print_prompt = self.verbose
        
        print(f"\n{'='*70}")
        print("Making Mobility Prediction")
        print(f"{'='*70}")
        
        # Display observation trajectory
        print(f"\nObservation Trajectory ({len(observation_trajectory)} points):")
        for i, point in enumerate(observation_trajectory, 1):
            loc_id = point.get('location_id', 'unknown')
            timestamp = point.get('timestamp', '')
            print(f"  {i}. Location {loc_id} at {timestamp}")
        
        current_location = observation_trajectory[-1]['location_id']
        print(f"\nCurrent Location: Grid {current_location}")
        
        if ground_truth is not None:
            print(f"Ground Truth Next Location: Grid {ground_truth}")
        
        # Step 1: Get RAG summary
        print(f"\n{'='*70}")
        print("Step 1: Retrieving Similar Trajectories (RAG)")
        print(f"{'='*70}")
        
        rag_summary, similar_samples, similarities = self.rag.generate_rag_summary(
            query_trajectory=observation_trajectory,
            top_m=self.top_m,
            print_prompt=print_prompt
        )
        
        print(f"\n✓ Retrieved {len(similar_samples)} similar samples")
        print(f"  Top similarity: {similarities[0]:.4f}")
        
        # Step 2: Get candidate locations from Gravity Model
        print(f"\n{'='*70}")
        print("Step 2: Generating Candidate Locations (Gravity Model)")
        print(f"{'='*70}")
        
        candidates_by_category = self.gravity.get_candidate_locations(
            current_grid_id=current_location,
            return_scores=True
        )
        
        print(f"\n✓ Generated {len(candidates_by_category)} POI category groups")
        print(f"  Each category has top-{self.top_n} candidates")
        
        # Display sample candidates
        print(f"\nSample Candidates (top 3 categories):")
        sample_categories = list(candidates_by_category.keys())[:3]
        for category in sample_categories:
            category_display = category.replace('_count', '')
            candidates = candidates_by_category[category]
            print(f"  {category_display}: {[c[0] for c in candidates[:3]]}")
        
        # Step 3: Make final prediction using LLM
        print(f"\n{'='*70}")
        print("Step 3: Final Prediction (LLM)")
        print(f"{'='*70}")
        
        predictions = self._generate_final_prediction(
            observation_trajectory=observation_trajectory,
            rag_summary=rag_summary,
            candidates_by_category=candidates_by_category,
            print_prompt=print_prompt
        )
        
        # Display predictions
        print(f"\n✓ Final Predictions (Top-{self.top_k}):")
        for i, (loc_id, confidence) in enumerate(predictions, 1):
            match_indicator = "✓" if ground_truth is not None and loc_id == ground_truth else " "
            print(f"  {match_indicator} {i}. Grid {loc_id} - Confidence: {confidence:.4f}")
        
        # Check if ground truth is in predictions
        if ground_truth is not None:
            predicted_locations = [loc_id for loc_id, _ in predictions]
            if ground_truth in predicted_locations:
                rank = predicted_locations.index(ground_truth) + 1
                print(f"\n✓ Ground truth found at rank {rank}")
            else:
                print(f"\n✗ Ground truth not in top-{self.top_k} predictions")
        
        print(f"{'='*70}\n")
        
        # Prepare results dictionary
        results = {
            'predictions': predictions,
            'rag_summary': rag_summary,
            'similar_samples': similar_samples,
            'similarities': similarities,
            'candidates_by_category': candidates_by_category,
            'current_location': current_location,
            'observation_trajectory': observation_trajectory
        }
        
        return predictions, results
    
    def _generate_final_prediction(
        self,
        observation_trajectory: List[Dict[str, Any]],
        rag_summary: str,
        candidates_by_category: Dict[str, List[Tuple[int, float]]],
        print_prompt: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate final prediction using LLM with RAG summary and gravity candidates.
        
        Args:
            observation_trajectory: Observation trajectory
            rag_summary: Summary from RAG retrieval
            candidates_by_category: Candidates grouped by POI category
            print_prompt: If True, print the prompt sent to LLM
        
        Returns:
            List of (location_id, confidence) tuples for top-K predictions
        """
        # Build comprehensive prompt
        prompt = self._build_final_prediction_prompt(
            observation_trajectory,
            rag_summary,
            candidates_by_category
        )
        
        # Print prompt if requested
        if print_prompt:
            print(f"\n{'='*70}")
            print("PROMPT SENT TO LLM (Final Prediction):")
            print(f"{'='*70}")
            print(prompt)
            print(f"{'='*70}\n")
        
        # Get all unique candidate locations
        all_candidates = set()
        for category, candidates in candidates_by_category.items():
            for grid_id, score in candidates:
                all_candidates.add(grid_id)
        
        candidate_list = list(all_candidates)
        
        # Tokenize and generate
        inputs = self.llm.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.llm.tokenizer.pad_token_id,
                eos_token_id=self.llm.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.llm.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()
        
        print(f"\nLLM Response:")
        print(f"{'-'*70}")
        print(response[:500] + "..." if len(response) > 500 else response)
        print(f"{'-'*70}")
        
        # Parse predictions from response
        predictions = self._parse_predictions_from_response(
            response,
            candidate_list,
            top_k=self.top_k
        )
        
        return predictions
    
    def _build_final_prediction_prompt(
        self,
        observation_trajectory: List[Dict[str, Any]],
        rag_summary: str,
        candidates_by_category: Dict[str, List[Tuple[int, float]]]
    ) -> str:
        """
        Build the final prediction prompt combining RAG summary and gravity candidates.
        
        Args:
            observation_trajectory: Observation trajectory
            rag_summary: Summary from RAG retrieval
            candidates_by_category: Candidates grouped by POI category
        
        Returns:
            Formatted prompt string
        """
        # Determine mobility mode
        if self.city == 'shenzhen':
            mobility_mode = "private car mobility"
        else:
            mobility_mode = "general mobility"
        
        prompt = f"You are a mobility prediction expert analyzing {mobility_mode} patterns. "
        prompt += "Your task is to predict the next location based on the following information:\n\n"
        
        # Add observation trajectory
        prompt += "## Current Trajectory:\n"
        for i, point in enumerate(observation_trajectory, 1):
            loc_id = point.get('location_id', 'unknown')
            timestamp = point.get('timestamp', '')
            prompt += f"{i}. Location {loc_id} at {timestamp}\n"
        
        # Add RAG summary
        prompt += f"\n## Similar Historical Patterns:\n"
        prompt += f"{rag_summary}\n"
        
        # Add gravity model candidates by category
        prompt += f"\n## Candidate Locations by POI Category:\n"
        prompt += "Based on the gravity model, here are the most attractive locations for each category:\n\n"
        
        # Show top 3 categories with their candidates
        category_count = 0
        for category, candidates in candidates_by_category.items():
            if category_count >= 5:  # Limit to top 5 categories to avoid prompt length
                break
            
            category_display = category.replace('_count', '')
            candidate_locs = [f"Grid {c[0]}" for c in candidates[:3]]
            prompt += f"- {category_display}: {', '.join(candidate_locs)}\n"
            category_count += 1
        
        # Get all unique candidates for the final instruction
        all_candidates = set()
        for category, candidates in candidates_by_category.items():
            for grid_id, score in candidates[:3]:  # Top 3 per category
                all_candidates.add(grid_id)
        
        candidate_list_str = ", ".join([f"Grid {c}" for c in sorted(list(all_candidates))[:20]])
        
        # Add prediction instruction
        prompt += f"\n## Your Task:\n"
        prompt += f"Based on the trajectory pattern, similar historical behaviors, and candidate locations, "
        prompt += f"predict the top {self.top_k} most likely next locations.\n"
        prompt += f"Choose from these candidates: {candidate_list_str}\n\n"
        prompt += f"Format your answer as: Grid [ID], Grid [ID], Grid [ID], ...\n"
        prompt += f"Provide exactly {self.top_k} predictions in order of likelihood.\n\n"
        prompt += "Your predictions:"
        
        return prompt
    
    def _parse_predictions_from_response(
        self,
        response: str,
        candidate_list: List[int],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Parse predictions from LLM response.
        
        Args:
            response: LLM response text
            candidate_list: List of candidate location IDs
            top_k: Number of predictions to return
        
        Returns:
            List of (location_id, confidence) tuples
        """
        import re
        
        # Extract grid IDs from response
        grid_pattern = r'Grid\s+(\d+)'
        matches = re.findall(grid_pattern, response, re.IGNORECASE)
        
        predictions = []
        seen = set()
        
        for match in matches:
            grid_id = int(match)
            if grid_id not in seen and grid_id in candidate_list:
                # Confidence decreases with rank
                confidence = 1.0 / (len(predictions) + 1)
                predictions.append((grid_id, confidence))
                seen.add(grid_id)
                
                if len(predictions) >= top_k:
                    break
        
        # If not enough predictions found, fill with top candidates
        if len(predictions) < top_k:
            for grid_id in candidate_list:
                if grid_id not in seen:
                    confidence = 1.0 / (len(predictions) + 1)
                    predictions.append((grid_id, confidence))
                    seen.add(grid_id)
                    
                    if len(predictions) >= top_k:
                        break
        
        return predictions[:top_k]
    
    def batch_predict(
        self,
        trajectories: List[List[Dict[str, Any]]],
        ground_truths: List[int] = None,
        print_prompt: bool = False
    ) -> List[Tuple[List[Tuple[int, float]], Dict[str, Any]]]:
        """
        Make predictions for multiple trajectories.
        
        Args:
            trajectories: List of observation trajectories
            ground_truths: List of ground truth next locations (optional)
            print_prompt: If True, print prompts
        
        Returns:
            List of (predictions, results) tuples for each trajectory
        """
        results = []
        
        if ground_truths is None:
            ground_truths = [None] * len(trajectories)
        
        for i, (traj, gt) in enumerate(zip(trajectories, ground_truths), 1):
            print(f"\n{'#'*70}")
            print(f"Prediction {i}/{len(trajectories)}")
            print(f"{'#'*70}")
            
            pred, result = self.predict(
                observation_trajectory=traj,
                ground_truth=gt,
                print_prompt=print_prompt
            )
            
            results.append((pred, result))
        
        return results
    
    def evaluate(
        self,
        trajectories: List[List[Dict[str, Any]]],
        ground_truths: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate prediction performance.
        
        Args:
            trajectories: List of observation trajectories
            ground_truths: List of ground truth next locations
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.batch_predict(
            trajectories=trajectories,
            ground_truths=ground_truths,
            print_prompt=False
        )
        
        # Calculate metrics
        hits_at_k = {k: 0 for k in [1, 3, 5]}
        mrr_sum = 0.0
        total = len(ground_truths)
        
        for (preds, _), gt in zip(predictions, ground_truths):
            predicted_locs = [loc_id for loc_id, _ in preds]
            
            # Check hits@k
            for k in hits_at_k.keys():
                if gt in predicted_locs[:k]:
                    hits_at_k[k] += 1
            
            # Calculate MRR
            if gt in predicted_locs:
                rank = predicted_locs.index(gt) + 1
                mrr_sum += 1.0 / rank
        
        # Compute final metrics
        metrics = {
            'accuracy@1': hits_at_k[1] / total,
            'accuracy@3': hits_at_k[3] / total,
            'accuracy@5': hits_at_k[5] / total,
            'mrr': mrr_sum / total,
            'total_samples': total
        }
        
        # Print metrics
        print(f"\n{'='*70}")
        print("Evaluation Results")
        print(f"{'='*70}")
        print(f"Total samples: {total}")
        print(f"Accuracy@1: {metrics['accuracy@1']:.4f}")
        print(f"Accuracy@3: {metrics['accuracy@3']:.4f}")
        print(f"Accuracy@5: {metrics['accuracy@5']:.4f}")
        print(f"MRR: {metrics['mrr']:.4f}")
        print(f"{'='*70}\n")
        
        return metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the predictor configuration."""
        return {
            'city': self.city,
            'top_k': self.top_k,
            'top_m': self.top_m,
            'top_n': self.top_n,
            'gravity_weight': self.gravity_weight,
            'gravity_radius': self.gravity_radius,
            'llm_model': self.llm.model_name,
            'rag_database_size': len(self.rag.database_samples) if self.rag.database_samples else 0,
            'poi_locations': len(self.gravity.poi_data)
        }
    
    def print_statistics(self):
        """Print predictor statistics."""
        stats = self.get_statistics()
        
        print(f"\n{'='*50}")
        print("MobilityPredictor Statistics")
        print(f"{'='*50}")
        print(f"City: {stats['city']}")
        print(f"LLM Model: {stats['llm_model']}")
        print(f"Top-K Predictions: {stats['top_k']}")
        print(f"RAG Top-M: {stats['top_m']}")
        print(f"Gravity Top-N: {stats['top_n']}")
        print(f"Gravity Weight: {stats['gravity_weight']}")
        print(f"Gravity Radius: {stats['gravity_radius']}")
        print(f"RAG Database Size: {stats['rag_database_size']}")
        print(f"POI Locations: {stats['poi_locations']}")
        print(f"{'='*50}\n")
