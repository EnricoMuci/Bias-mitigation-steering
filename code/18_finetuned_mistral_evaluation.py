#!/usr/bin/env python3
"""
Fine-tuned Mistral Evaluation Script

Evaluates the fine-tuned Mistral model on bias benchmarks without steering vectors.
Uses dummy vectors with coefficient 0 to maintain compatibility with evaluation functions.
"""

import sys
import os
import pandas as pd
import logging
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Import functions from the individual evaluation files
from importlib import import_module

bbq_eval = import_module('6_bbq_evaluation')
mmlu_eval = import_module('7_mmlu_evaluation')
stereoset_eval = import_module('8_stereoset_evaluation')
crows_eval = import_module('9_crows_pairs_evaluation')
clear_bias_eval = import_module('10_clear_bias_evaluation')


class DummySteeringModel:
    """Wrapper for the fine-tuned model to match SteeringModel interface."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        
        # Load fine-tuned adapter
        self.model = PeftModel.from_pretrained(self.model, model_path)
        
        # Set device attribute for compatibility
        self.device = next(self.model.parameters()).device
        
    def half(self):
        """For compatibility with evaluation functions."""
        return self
        
    def generate(self, input_ids, **kwargs):
        """Generate text using the fine-tuned model."""
        return self.model.generate(input_ids, **kwargs)
        
    def __call__(self, *args, **kwargs):
        """For compatibility with evaluation functions."""
        return self.model(*args, **kwargs)
    
    def to(self, device):
        """For compatibility with evaluation functions."""
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def eval(self):
        """For compatibility with evaluation functions."""
        self.model.eval()
        return self
    
    def parameters(self):
        """For compatibility with evaluation functions."""
        return self.model.parameters()
    
    def named_parameters(self):
        """For compatibility with evaluation functions."""
        return self.model.named_parameters()
    
    @property
    def config(self):
        """For compatibility with evaluation functions."""
        return self.model.config
    
    def set_control(self, vector, coeff, **kwargs):
        """Dummy set_control method for compatibility with evaluation functions.
        
        Since this is a fine-tuned model, we don't actually apply steering vectors.
        The steering effect is already built into the fine-tuned weights.
        """
        # Do nothing - fine-tuned model already has the desired behavior
        pass
    
    def reset(self):
        """Dummy reset method for compatibility with evaluation functions."""
        # Do nothing for fine-tuned model
        pass


class DummySteeringVector:
    """Dummy steering vector with coefficient 0."""
    
    def __init__(self, dim=4096):
        # Create a zero vector 
        self.vector = np.zeros(dim, dtype=np.float32)
        
    def to_tensor(self):
        """Convert to tensor for compatibility."""
        return torch.zeros(4096, dtype=torch.float16)


def run_finetuned_evaluations():
    """Run all evaluations on the fine-tuned Mistral model."""
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../logs/finetuned_mistral_evaluation_{timestamp}.log"
    os.makedirs("../logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting fine-tuned Mistral evaluation...")
    
    # Model and tokenizer setup
    model_path = "../models/mistral-full-bbq-finetuned"
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load fine-tuned model
    logger.info("Loading fine-tuned model...")
    model = DummySteeringModel(model_path)
    
    # Create dummy vector
    dummy_vector = DummySteeringVector()
    coeff = 0.0  # No steering applied
    
    # Test configurations (one for each bias type to get comprehensive results)
    test_configs = [
        {'axis': 'gender', 'vector_type': 'finetuned', 'layer': 15, 'coeff': 0.0},
        {'axis': 'race', 'vector_type': 'finetuned', 'layer': 15, 'coeff': 0.0},
        {'axis': 'religion', 'vector_type': 'finetuned', 'layer': 15, 'coeff': 0.0},
        {'axis': 'age', 'vector_type': 'finetuned', 'layer': 15, 'coeff': 0.0},
        {'axis': 'nationality', 'vector_type': 'finetuned', 'layer': 15, 'coeff': 0.0},
        {'axis': 'socioeconomic', 'vector_type': 'finetuned', 'layer': 15, 'coeff': 0.0},
        {'axis': 'appearance', 'vector_type': 'finetuned', 'layer': 15, 'coeff': 0.0},
        {'axis': 'disability', 'vector_type': 'finetuned', 'layer': 15, 'coeff': 0.0}
    ]
    
    results = []
    
    for config in test_configs:
        axis = config['axis']
        logger.info(f"Processing axis: {axis}")
        
        result_row = {
            'axis': axis,
            'vector_type': config['vector_type'],
            'layer': config['layer'],
            'coeff': config['coeff'],
            'model_type': 'finetuned'
        }
        
        # Run BBQ evaluation
        try:
            logger.info(f"  Running BBQ evaluation for {axis}...")
            bbq_result = bbq_eval.run_bbq_evaluation(model, dummy_vector, coeff, axis, tokenizer)
            if bbq_result:
                for key, value in bbq_result.items():
                    if key != 'axis':
                        result_row[f'bbq_{key}'] = value
            logger.info(f"  BBQ evaluation completed for {axis}")
        except Exception as e:
            logger.error(f"  Error in BBQ evaluation for {axis}: {e}")
            result_row['bbq_error'] = str(e)
        
        # Run MMLU evaluation
        try:
            logger.info(f"  Running MMLU evaluation for {axis}...")
            mmlu_result = mmlu_eval.run_mmlu_evaluation(model, dummy_vector, coeff, axis, tokenizer)
            if mmlu_result:
                for key, value in mmlu_result.items():
                    if key not in ['axis', 'coeff']:
                        result_row[f'mmlu_{key}'] = value
            logger.info(f"  MMLU evaluation completed for {axis}")
        except Exception as e:
            logger.error(f"  Error in MMLU evaluation for {axis}: {e}")
            result_row['mmlu_error'] = str(e)
        
        # Bias evaluations based on relevance
        bias_evaluations = [
            ("StereoSet", stereoset_eval.run_stereoset_evaluation, ['gender', 'race', 'religion'], 'stereoset'),
            ("CrowS-Pairs", crows_eval.run_crows_pairs_evaluation, ['gender', 'race', 'religion', 'age', 'nationality', 'socioeconomic', 'appearance', 'disability'], 'crows'),
            ("Clear Bias", clear_bias_eval.run_clear_bias_evaluation, ['gender', 'race', 'age', 'disability', 'religion', 'socioeconomic'], 'clear_bias')
        ]
        
        for eval_name, eval_func, relevant_axes, prefix in bias_evaluations:
            if axis in relevant_axes:
                try:
                    logger.info(f"  Running {eval_name} evaluation for {axis}...")
                    eval_result = eval_func(model, dummy_vector, coeff, axis, tokenizer)
                    if eval_result:
                        for key, value in eval_result.items():
                            if key not in ['axis', 'coeff']:
                                result_row[f'{prefix}_{key}'] = value
                    logger.info(f"  {eval_name} evaluation completed for {axis}")
                except Exception as e:
                    logger.error(f"  Error in {eval_name} evaluation for {axis}: {e}")
                    result_row[f'{prefix}_error'] = str(e)
            else:
                logger.info(f"  Skipping {eval_name} evaluation for {axis} (not relevant)")
                result_row[f'{prefix}_skipped'] = True
        
        results.append(result_row)
    
    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs("../results/mistral-full", exist_ok=True)
    results_file = f"../results/mistral-full/finetuned_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    
    logger.info("="*60)
    logger.info("Fine-tuned Mistral evaluation completed!")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Log saved to: {log_file}")
    logger.info(f"Evaluated {len(results)} configurations")
    logger.info("="*60)


if __name__ == "__main__":
    run_finetuned_evaluations()