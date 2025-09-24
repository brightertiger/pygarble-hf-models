"""
Sentence Transformer Classifier

A production-ready library for fine-tuning sentence transformers for binary text classification.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .training.classifier import BERTBinaryClassifier
from .training.dataset import TextClassificationDataset, load_data_from_csv

__all__ = [
    "BERTBinaryClassifier",
    "TextClassificationDataset", 
    "load_data_from_csv"
]
