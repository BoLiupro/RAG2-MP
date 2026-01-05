"""
MobilityPredictor Module
This module integrates LLM+RAG and Gravity Model for mobility prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from .LLM import MobilityLLM
from .RAG import MobilityRAG
from .Gravity import GravityModel
from util.utils import format_trajectory_with_distances, calculate_grid_distance, format_candidates_with_distances


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
        rag_database_path: str = "/workspace/China_Journal/util/rag_database",
        city: str = "beijing",
        top_k_predictions: int = 5,
        rag_top_m_samples: int = 5,
        gravity_top_n_candidates: int = 5,
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
            city: City name
            top_k_predictions: Number of top predictions to return
            rag_top_m_samples: Number of similar samples to retrieve (RAG)
            gravity_top_n_candidates: Number of candidates per POI category (Gravity)
            gravity_weight: Weight parameter for gravity model
            gravity_radius: Search radius for gravity model (in grid units)
            use_quantization: Whether to use 4-bit quantization for LLM
            use_lora: Whether to use LoRA fine-tuning
            verbose: If True, print prompts and intermediate results
        """
        self.city = city
        self.top_k_predictions = top_k_predictions
        self.rag_top_m_samples = rag_top_m_samples
        self.gravity_top_n_candidates = gravity_top_n_candidates
        self.gravity_weight = gravity_weight
        self.gravity_radius = gravity_radius
        self.verbose = verbose
        self.num_grids = 1600  # Standard grid size for all cities
        
        # Initialize LLM
        self.llm = MobilityLLM(
            model_name=llm_model_name,
            model_path=llm_model_path,
            use_quantization=use_quantization,
            use_lora=use_lora
        )
        
        # Initialize RAG module
        self.rag = MobilityRAG(
            llm=self.llm,
            rag_database_path=rag_database_path,
            rag_top_m_samples=rag_top_m_samples,
            city=city,
            verbose=verbose
        )
        
        # Load RAG database
        self.rag.load_rag_database()
        
        # POI data path is always data/{city}/poi.csv
        poi_data_path = f"/workspace/China_Journal/data/{city}/poi.csv"
        
        # Load POI data for RAG
        self.rag.load_poi_data(poi_data_path)
        
        # Initialize Gravity Model
        self.gravity = GravityModel(
            poi_data_path=poi_data_path,
            city=city,
            weight=gravity_weight,
            gravity_top_n_candidates=gravity_top_n_candidates,
            radius=gravity_radius
        )
    
    def predict(
        self,
        observation_trajectory: List[Dict[str, Any]],
        ground_truth: int = None,
        print_prompt: bool = None,
        use_beam_search: bool = False
    ) -> Tuple[List[Tuple[int, float]], torch.Tensor, List[int]]:
        """
        Make mobility prediction for the next location using LLM generation.
        
        Args:
            observation_trajectory: List of trajectory points, each with 'location_id' and 'timestamp'
            ground_truth: Ground truth next location (optional, for evaluation)
            print_prompt: If True, print the final prediction prompt (overrides self.verbose)
            use_beam_search: If True, use beam search for multiple predictions (testing)
        
        Returns:
            Tuple of:
                - List of (location_id, confidence) tuples for top-K predictions
                - Logits tensor (or None for generation-based approach)
                - List of candidate grid IDs
        """
        if print_prompt is None:
            print_prompt = self.verbose
        
        current_location = observation_trajectory[-1]['location_id']
        
        # Step 1: Get RAG summary
        rag_summary, similar_samples, similarities = self.rag.generate_rag_summary(
            query_trajectory=observation_trajectory,
            rag_top_m_samples=self.rag_top_m_samples,
            print_prompt=print_prompt
        )
        
        # Step 2: Get candidate locations from Gravity Model
        candidates_by_category = self.gravity.get_candidate_locations(
            current_grid_id=current_location,
            return_scores=True
        )
        
        # Step 3: Make final prediction using LLM generation
        predictions, logits, candidate_list = self._generate_final_prediction_with_classifier(
            observation_trajectory=observation_trajectory,
            rag_summary=rag_summary,
            candidates_by_category=candidates_by_category,
            print_prompt=print_prompt,
            use_beam_search=use_beam_search
        )
        
        # Prepare results dictionary
        results = {
            'predictions': predictions,
            'rag_summary': rag_summary,
            'similar_samples': similar_samples,
            'similarities': similarities,
            'candidates_by_category': candidates_by_category,
            'current_location': current_location,
            'observation_trajectory': observation_trajectory,
            'candidate_list': candidate_list,
            'logits': logits  # May be None for generation-based approach
        }
        
        return predictions, logits, candidate_list
    
    def _generate_final_prediction_with_classifier(
        self,
        observation_trajectory: List[Dict[str, Any]],
        rag_summary: str,
        candidates_by_category: Dict[str, List[Tuple[int, float]]],
        print_prompt: bool = True,
        use_beam_search: bool = False
    ) -> Tuple[List[Tuple[int, float]], torch.Tensor, List[int]]:
        """
        Generate final prediction using LLM generation (with or without beam search).
        
        Args:
            observation_trajectory: Observation trajectory
            rag_summary: Summary from RAG retrieval
            candidates_by_category: Candidates grouped by POI category
            print_prompt: If True, print the prompt sent to LLM
            use_beam_search: If True, use beam search for multiple predictions
        
        Returns:
            Tuple of (predictions, logits, candidate_list)
            - predictions: List of (location_id, confidence) tuples
            - logits: None (no logits for generation-based approach)
            - candidate_list: List of all grid IDs
        """
        # Build comprehensive prompt
        prompt = self._build_final_prediction_prompt(
            observation_trajectory,
            rag_summary,
            candidates_by_category
        )
        
        # Get all unique candidate locations
        all_candidates = set()
        for category, candidates in candidates_by_category.items():
            for grid_id, score in candidates:
                all_candidates.add(grid_id)
        
        candidate_list = sorted(list(all_candidates))
        
        # Tokenize prompt
        inputs = self.llm.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.llm.device)
        
        if use_beam_search:
            # Testing mode: Use beam search for multiple predictions
            with torch.no_grad():
                outputs = self.llm.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    num_beams=self.top_k_predictions,
                    num_return_sequences=self.top_k_predictions,
                    diversity_penalty=1.0,
                    early_stopping=True,
                    pad_token_id=self.llm.tokenizer.pad_token_id,
                    eos_token_id=self.llm.tokenizer.eos_token_id
                )
            
            # Decode all beam search results
            predictions = []
            for i, output in enumerate(outputs):
                generated_text = self.llm.tokenizer.decode(output, skip_special_tokens=True)
                
                # Extract generated part (remove prompt)
                if generated_text.startswith(prompt):
                    response = generated_text[len(prompt):].strip()
                else:
                    response = generated_text.strip()
                
                # Parse grid_id from response
                grid_id = self._parse_grid_id_from_response(response)
                
                if grid_id is not None and 0 <= grid_id < self.num_grids:
                    # Confidence decreases with beam rank
                    confidence = 1.0 / (i + 1)
                    predictions.append((grid_id, confidence))
            
            # Fill with candidates if needed
            seen_grids = {grid_id for grid_id, _ in predictions}
            for grid_id in candidate_list:
                if grid_id not in seen_grids and len(predictions) < self.top_k_predictions:
                    confidence = 1.0 / (len(predictions) + 1)
                    predictions.append((grid_id, confidence))
                    seen_grids.add(grid_id)
        
        else:
            # Training mode: Greedy generation (single output)
            with torch.no_grad():
                outputs = self.llm.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=self.llm.tokenizer.pad_token_id,
                    eos_token_id=self.llm.tokenizer.eos_token_id
                )
            
            generated_text = self.llm.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            # Parse grid_id from response
            grid_id = self._parse_grid_id_from_response(response)
            
            if grid_id is not None and 0 <= grid_id < self.num_grids:
                predictions = [(grid_id, 1.0)]
            else:
                # Fallback to first candidate
                predictions = [(candidate_list[0], 1.0)] if candidate_list else [(0, 1.0)]
        
        # Return predictions (no logits needed for generation-based approach)
        candidate_list = list(range(self.num_grids))
        return predictions, None, candidate_list
    
    def _parse_grid_id_from_response(self, response: str) -> int:
        """
        Parse grid_id from LLM response.
        Expected format: "Grid 123" or "123"
        
        Args:
            response: LLM response text
        
        Returns:
            grid_id as integer, or None if parsing fails
        """
        import re
        
        # Try pattern "Grid XXX"
        match = re.search(r'Grid\s+(\d+)', response, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Try just digits
        match = re.search(r'(\d+)', response)
        if match:
            return int(match.group(1))
        
        return None
    
    def _build_prediction_prompt(
        self,
        observation_trajectory: List[Dict[str, Any]],
        rag_summary: str,
        candidates_by_category: Dict[str, List[Tuple[int, float]]]
    ) -> str:
        """
        Public alias for _build_final_prediction_prompt for compatibility.
        
        Args:
            observation_trajectory: Observation trajectory
            rag_summary: Summary from RAG retrieval
            candidates_by_category: Candidates grouped by POI category
        
        Returns:
            Formatted prompt string
        """
        return self._build_final_prediction_prompt(
            observation_trajectory,
            rag_summary,
            candidates_by_category
        )
    
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
        from util.utils import get_mobility_mode
        mobility_mode = get_mobility_mode(self.city)
        
        prompt = f"You are a mobility prediction expert analyzing {mobility_mode} patterns. "
        prompt += "Note: Trajectories may include both movement and stationary periods (staying at the same location). "
        prompt += "Your task is to predict the next location based on the following information:\n\n"
        
        # Add observation trajectory with distances
        prompt += "## Current Trajectory:\n"
        formatted_traj = format_trajectory_with_distances(
            observation_trajectory,
            city=self.city,
            include_poi=False,
            poi_data=None
        )
        prompt += formatted_traj + "\n"
        
        # Add RAG summary
        prompt += f"\n## Next location of similar mobility trajectories:\n"
        prompt += f"{rag_summary}\n"
        
        # Add gravity model candidates by category with distances
        prompt += f"\n## Candidate Locations by POI Category:\n"
        prompt += "Based on the gravity model, here are the most attractive locations for each category:\n"
        prompt += "(Note: Current location is included as a candidate for stationary behavior)\n\n"
        
        current_location = observation_trajectory[-1]['location_id']
        
        # Show top 5 categories with their candidates and distances
        category_count = 0
        for category, candidates in candidates_by_category.items():
            # if category_count >= 5:  # Limit to top 5 categories to avoid prompt length
            #     break
            
            category_display = category.replace('_count', '')
            candidate_str = format_candidates_with_distances(
                candidates,
                current_location,
                city=self.city,
                include_scores=False
            )
            prompt += f"- {category_display}: {candidate_str}\n"
            category_count += 1
        
        # Get all unique candidates for the final instruction
        all_candidates = set()
        for category, candidates in candidates_by_category.items():
            for grid_id, score in candidates[:3]:  # Top 3 per category
                all_candidates.add(grid_id)
        
        candidate_list_str = ", ".join([f"Grid {c}" for c in sorted(list(all_candidates))[:20]])
        
        # Add prediction instruction - simplified for single grid output
        prompt += f"\n## Your Task:\n"
        prompt += f"Based on the trajectory pattern, similar historical behaviors, and candidate locations, "
        prompt += f"predict the next most likely location.\n"
        prompt += f"Remember: The user may stay at the current location (Grid {current_location}) or move to a new location.\n\n"
        prompt += f"Output format: Grid [ID]\n"
        prompt += f"Output only the grid ID, no explanation.\n\n"
        prompt += "Your prediction:"
        
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the predictor configuration."""
        return {
            'city': self.city,
            'top_k_predictions': self.top_k_predictions,
            'rag_top_m_samples': self.rag_top_m_samples,
            'gravity_top_n_candidates': self.gravity_top_n_candidates,
            'gravity_weight': self.gravity_weight,
            'gravity_radius': self.gravity_radius,
            'llm_model': self.llm.model_name,
            'rag_database_size': len(self.rag.database_samples) if self.rag.database_samples else 0,
            'poi_locations': len(self.gravity.poi_data)
        }
