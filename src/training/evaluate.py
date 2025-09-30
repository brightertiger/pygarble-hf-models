#!/usr/bin/env python3
"""
Evaluation script for the trained binary classification model.
"""

import argparse
import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer
import pytorch_lightning as pl

from src.training.classifier import BERTBinaryClassifier
from src.training.dataset import load_data_from_csv, create_data_loaders
from src.training.utils import (
    load_config, get_device, plot_confusion_matrix, 
    print_classification_report, save_predictions
)


def evaluate_model(
    checkpoint_path: str,
    test_data_path: str,
    config_path: str,
    output_dir: str = "evaluation_results"
):
    """Evaluate the trained model on test data."""
    
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = BERTBinaryClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load test data
    print("Loading test data...")
    test_texts, test_labels = load_data_from_csv(
        test_data_path,
        config['data']['text_column'],
        config['data']['label_column']
    )
    
    print(f"Test samples: {len(test_texts)}")
    
    # Create test data loader
    _, test_loader = create_data_loaders(
        test_texts, test_labels, test_texts, test_labels,
        tokenizer,
        config['training']['batch_size'],
        config['model']['max_length'],
        config['hardware']['num_workers']
    )
    
    # Setup trainer for evaluation
    trainer = pl.Trainer(
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        logger=False
    )
    
    # Run evaluation
    print("Running evaluation...")
    results = trainer.test(model, test_loader)
    
    # Get predictions
    print("Generating predictions...")
    predictions = []
    probabilities = []
    
    model.to(get_device())
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(get_device())
            attention_mask = batch['attention_mask'].to(get_device())
            
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Convert labels to numpy array
    test_labels = [int(label) for label in test_labels]
    
    # Print classification report
    print_classification_report(test_labels, predictions)
    
    # Plot confusion matrix
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(test_labels, predictions, save_path=confusion_matrix_path)
    
    # Save predictions
    predictions_path = os.path.join(output_dir, "predictions.csv")
    save_predictions(predictions, test_labels, test_texts, predictions_path)
    
    # Save detailed results
    results_path = os.path.join(output_dir, "detailed_results.csv")
    detailed_df = pd.DataFrame({
        'text': test_texts,
        'true_label': test_labels,
        'predicted_label': predictions,
        'probability_normal': [prob[0] for prob in probabilities],
        'probability_garbled': [prob[1] for prob in probabilities],
        'correct': [true == pred for true, pred in zip(test_labels, predictions)]
    })
    detailed_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to {results_path}")
    
    # Print summary statistics
    accuracy = sum(detailed_df['correct']) / len(detailed_df)
    print(f"\nEvaluation Summary:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Total Samples: {len(test_texts)}")
    print(f"Correct Predictions: {sum(detailed_df['correct'])}")
    print(f"Incorrect Predictions: {len(test_texts) - sum(detailed_df['correct'])}")
    
    return results, detailed_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained binary classification model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-data", required=True, help="Path to test data CSV file")
    parser.add_argument("--config", default="src/training/config.yaml", help="Path to config file")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.test_data):
        print(f"Error: Test data file not found: {args.test_data}")
        sys.exit(1)
    
    try:
        results, detailed_df = evaluate_model(
            args.checkpoint,
            args.test_data,
            args.config,
            args.output_dir
        )
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
