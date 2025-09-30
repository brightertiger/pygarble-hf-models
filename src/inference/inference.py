#!/usr/bin/env python3
"""
Model inference module for binary text classification.
Handles model loading and single/batch predictions.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Any
import logging

class ModelInference:
    """Lightweight model inference class without PyTorch Lightning."""
    
    def __init__(self, checkpoint_path: str, model_name: str = "huawei-noah/TinyBERT_General_4L_312D", max_length: int = 512, device: str = "auto"):
        self.logger = logging.getLogger(__name__)
        self.max_length = max_length
        
        # Setup device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        from src.training.classifier import BERTBinaryClassifier
        
        self.logger.info(f"Loading model from {checkpoint_path}")
        
        self.model = BERTBinaryClassifier.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device
        )
        
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("Model loaded successfully")
    
    def predict_single(self, text: str) -> Tuple[int, float, List[float]]:
        """Make prediction for a single text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            
            # Get probabilities
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probs).item()
            probabilities = probs.cpu().numpy()[0].tolist()
        
        return predicted_class, confidence, probabilities
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[int, float, List[float]]]:
        """Make predictions for a batch of texts.
        
        Args:
            texts: List of input texts to classify
            
        Returns:
            List of tuples (predicted_class, confidence, probabilities)
        """
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results
    
    def predict_with_timing(self, text: str) -> Tuple[int, float, List[float], float]:
        """Make prediction with timing information.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities, inference_time_seconds)
        """
        import time
        start_time = time.time()
        pred_class, confidence, probs = self.predict_single(text)
        inference_time = time.time() - start_time
        return pred_class, confidence, probs, inference_time
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'max_length': self.tokenizer.model_max_length
        }
