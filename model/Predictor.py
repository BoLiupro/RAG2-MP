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
        gravity_weight_config_path: str = None,
        use_quantization: bool = True,
        use_lora: bool = False,
        verbose: bool = True,
        # Generation parameters from config
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: int = 40,
        do_sample: bool = True,
        max_new_tokens: int = 256,
        # Prediction parameters
        prediction_time_interval: str = "1 hour"
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
            gravity_weight: Weight parameter for gravity model (default if no config)
            gravity_radius: Search radius for gravity model (in grid units)
            gravity_weight_config_path: Path to fitted weights JSON (optional)
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
        self.prediction_time_interval = prediction_time_interval
        
        # Generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        
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
            verbose=verbose,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens
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
            radius=gravity_radius,
            weight_config_path=gravity_weight_config_path
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
            truncation=False
            # max_length=1024
        ).to(self.llm.device)

        
        if use_beam_search:
            # Testing mode: Use beam search with sampling for diversity
            with torch.no_grad():
                outputs = self.llm.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.top_k_predictions,
                    num_return_sequences=self.top_k_predictions,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    early_stopping=False,
                    # repetition_penalty=1.2,
                    # no_repeat_ngram_size=2,  # Prevent repeating 2-grams
                    pad_token_id=self.llm.tokenizer.pad_token_id,
                    # eos_token_id=self.llm.tokenizer.eos_token_id
                )
            
            # Decode all beam outputs
            predictions = []
            seen_grids = set()
            
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
                    # Only add if not seen (for diversity)
                    if grid_id not in seen_grids:
                        confidence = 1.0 / (len(predictions) + 1)
                        predictions.append((grid_id, confidence))
                        seen_grids.add(grid_id)
            
            # Fill with candidates if we don't have enough unique predictions
            if len(predictions) < self.top_k_predictions:
                for grid_id in candidate_list:
                    if grid_id not in seen_grids:
                        confidence = 1.0 / (len(predictions) + 1)
                        predictions.append((grid_id, confidence))
                        seen_grids.add(grid_id)
                        if len(predictions) >= self.top_k_predictions:
                            break
        
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
        match = re.search(r'\n\nGrid\s+(\d+)', response, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Try just digits
        match = re.search(r'(\d+)', response)
        if match:
            return int(match.group(1))
        
        return None
    
    def _format_trajectory_compact(self, observation_trajectory: List[Dict[str, Any]]) -> str:
        """
        Format trajectory showing all steps with distances and top 2 POI categories.
        This detailed format helps LLM understand functional attributes and temporal patterns.
        
        Format: Grid 804 (Shopping & Consumer Goods, Life Services) ->0.20km-> Grid 804 (Shopping & Consumer Goods, Life Services) ->6.04km-> Grid 1216 (Companies & Enterprises, Residential)...
        
        Args:
            observation_trajectory: List of trajectory points with 'location_id' and 'timestamp'
        Returns:
            Detailed trajectory string with all steps, distances, and top 2 POI categories
        """
        if not observation_trajectory:
            return "Empty trajectory"

        from util.utils import calculate_grid_distance

        # Get POI data for area type information
        poi_data = self.rag.poi_data if hasattr(self.rag, 'poi_data') and self.rag.poi_data else {}

        def get_top_2_area_types(loc_id):
            """Get top 2 area types for a location"""
            if loc_id not in poi_data:
                return "Mixed"
            
            poi_features = poi_data[loc_id]
            # Get all POI types with their percentages
            poi_list = [(poi_type.replace('_', ' ').strip(), pct) 
                       for poi_type, pct in poi_features.items() if pct > 0]
            
            # Sort by percentage descending
            poi_list.sort(key=lambda x: x[1], reverse=True)
            
            # Get top 2 types
            top_2 = [name for name, pct in poi_list[:2]]
            
            if not top_2:
                return "Mixed"
            elif len(top_2) == 1:
                return top_2[0]
            else:
                return ", ".join(top_2)

        # Format each step in the trajectory
        formatted_parts = []
        
        for idx, point in enumerate(observation_trajectory):
            loc_id = point['location_id']
            area_types = get_top_2_area_types(loc_id)
            
            # Format this grid
            grid_str = f"Grid {loc_id} ({area_types})"
            
            if idx == 0:
                # First grid, no distance prefix
                formatted_parts.append(grid_str)
            else:
                # Calculate distance from previous grid
                prev_loc_id = observation_trajectory[idx - 1]['location_id']
                dist = calculate_grid_distance(prev_loc_id, loc_id, self.city, 40)
                formatted_parts.append(f" ->{dist:.2f}km-> {grid_str}")

        return "".join(formatted_parts)
    
    def _format_candidates_compact(self, candidates_by_category: Dict[str, List[Tuple[int, float]]],
                                   current_location: int) -> str:
        """
        Format candidate locations in a compact format by category.
        Each POI category displayed on a single line with attractive scores.
        
        Args:
            candidates_by_category: Dict mapping POI categories to candidate lists
            current_location: Current grid ID
        
        Returns:
            Compact formatted candidates string
        """
        from util.utils import calculate_grid_distance
        
        formatted_lines = []
        formatted_lines.append("Format: [Category]: [Grid ID] (Attractive Score,0-1), [Distance to current location]...")
        formatted_lines.append("(Locations are ranked by attractiveness; higher scores indicate more likely destinations)")
        
        # Sort categories and process all of them
        for category, candidates in sorted(candidates_by_category.items()):
            # Clean up category name and fix spelling
            category_display = category.replace('_count', '').replace('_', ' ').title()
            
            # Format all candidates for this category on a single line
            candidate_strs = []
            for grid_id, score in candidates[:self.gravity_top_n_candidates]:  # Strictly limit by config
                dist = calculate_grid_distance(current_location, grid_id, self.city, 40)
                candidate_strs.append(f"Grid {grid_id} ({score:.2f}), {dist:.2f}km")
            
            # Join all candidates with " | " separator
            formatted_lines.append(f"{category_display}: {' | '.join(candidate_strs)}")
        
        return "\n".join(formatted_lines)
    
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
        Uses compact, structured formatting for clarity.
        
        Args:
            observation_trajectory: Observation trajectory
            rag_summary: Summary from RAG retrieval
            candidates_by_category: Candidates grouped by POI category
        
        Returns:
            Formatted prompt string
        """
        
        current_location = observation_trajectory[-1]['location_id']

        prompt = f"""You must follow these rules:
- Do NOT explain.
- Do NOT think aloud.
- Do NOT output anything except the answer.
- Output the answer in this example format: Grid:21
"""
        prompt += f"## Your Task\n"
        prompt += f"You are a mobility prediction expert analyzing human mobility patterns.\n"
        prompt += f"The user may stay at Grid {current_location} or move to a new location.\n"
        prompt += f"Predict the next most likely location within the next {self.prediction_time_interval} based on the following information:\n\n"
        
        # Add detailed trajectory format
        prompt += "## Current Trajectory\n"
        prompt += "Format: Grid ID (Top 2 Area Types) with distances between consecutive locations\n"
        detailed_traj = self._format_trajectory_compact(observation_trajectory)
        prompt += detailed_traj + "\n\n"
        
        # Add RAG summary
        # prompt += "## Feature of next location of similar group mobility\n"
        # prompt += f"{rag_summary}\n\n"
        
        # Add compact candidate locations
        prompt += "## Candidate Locations\n"
        prompt += "(Current location included as a candidate for stationary behavior)\n"
        compact_candidates = self._format_candidates_compact(candidates_by_category, current_location)
        prompt += compact_candidates + "\n"

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
