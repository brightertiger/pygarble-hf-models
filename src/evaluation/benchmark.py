#!/usr/bin/env python3
"""
Evaluation script for fine-tuned sentence transformer models.
"""

import yaml
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

from src.core.classifier import SentenceTransformerClassifier
from src.core.dataset import load_data_from_csv, create_data_loaders


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(
    model_path: str,
    test_file: str,
    config: dict,
    output_dir: str = "evaluation_results"
):
    """
    Evaluate the model on test data and generate comprehensive metrics.
    
    Args:
        model_path: Path to the model checkpoint
        test_file: Path to test data CSV file
        config: Configuration dictionary
        output_dir: Directory to save evaluation results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load test data
    test_texts, test_labels = load_data_from_csv(
        test_file,
        config['data']['text_column'],
        config['data']['label_column']
    )
    
    print(f"Loaded {len(test_texts)} test samples")
    
    # Create test data loader
    _, test_loader = create_data_loaders(
        test_texts, test_labels, test_texts, test_labels,
        tokenizer,
        config['training']['batch_size'],
        config['model']['max_length'],
        config['hardware']['num_workers']
    )
    
    # Load model
    model = SentenceTransformerClassifier.load_from_checkpoint(model_path)
    model.eval()
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Collect predictions
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels)
    
    print(f"\\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    class_names = ['Negative', 'Positive']  # Adjust based on your labels
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    print("\\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'text': test_texts,
        'true_label': all_labels,
        'predicted_label': all_predictions,
        'confidence': all_probabilities.max(axis=1),
        'prob_negative': all_probabilities[:, 0],
        'prob_positive': all_probabilities[:, 1]
    })
    
    results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    # Save summary metrics
    summary_metrics = {
        'accuracy': accuracy,
        'precision_0': report['0']['precision'],
        'recall_0': report['0']['recall'],
        'f1_0': report['0']['f1-score'],
        'precision_1': report['1']['precision'],
        'recall_1': report['1']['recall'],
        'f1_1': report['1']['f1-score'],
        'macro_avg_precision': report['macro avg']['precision'],
        'macro_avg_recall': report['macro avg']['recall'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_precision': report['weighted avg']['precision'],
        'weighted_avg_recall': report['weighted avg']['recall'],
        'weighted_avg_f1': report['weighted avg']['f1-score']
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        import json
        json.dump(summary_metrics, f, indent=2)
    
    print(f"\\nEvaluation results saved to: {output_dir}")
    
    return summary_metrics


def predict_single_text(
    model_path: str,
    text: str,
    config: dict
):
    """
    Make prediction on a single text.
    
    Args:
        model_path: Path to the model checkpoint
        text: Input text
        config: Configuration dictionary
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load model
    model = SentenceTransformerClassifier.load_from_checkpoint(model_path)
    model.eval()
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=config['model']['max_length']
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_names = ['Negative', 'Positive']
    
    result = {
        'text': text,
        'predicted_class': predicted_class,
        'predicted_label': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': {
            'negative': probabilities[0][0].item(),
            'positive': probabilities[0][1].item()
        }
    }
    
    return result


def main():
    # Get model checkpoint from args (required)
    model_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--model' and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            break
    
    if not model_path:
        print("Error: --model argument is required")
        print("Usage: ./evaluate --model <checkpoint_path> [--config <config_file>] [--test-file <test_file>] [--text <text>] [--output-dir <output_dir>]")
        return
    
    # Get config file from args or use default
    config_file = 'src/configs/default.yaml'
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            break
    
    # Get test file from args
    test_file = None
    for i, arg in enumerate(sys.argv):
        if arg == '--test-file' and i + 1 < len(sys.argv):
            test_file = sys.argv[i + 1]
            break
    
    # Get text from args
    text = None
    for i, arg in enumerate(sys.argv):
        if arg == '--text' and i + 1 < len(sys.argv):
            text = sys.argv[i + 1]
            break
    
    # Get output directory from args or use default
    output_dir = 'evaluation_results'
    for i, arg in enumerate(sys.argv):
        if arg == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            break
    
    # Load configuration
    config = load_config(config_file)
    
    if text:
        # Single text prediction
        result = predict_single_text(model_path, text, config)
        print("\\nPrediction Result:")
        print(f"Text: {result['text']}")
        print(f"Predicted Class: {result['predicted_label']} ({result['predicted_class']})")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities:")
        print(f"  Negative: {result['probabilities']['negative']:.4f}")
        print(f"  Positive: {result['probabilities']['positive']:.4f}")
    
    elif test_file:
        # Full evaluation
        metrics = evaluate_model(model_path, test_file, config, output_dir)
        print("\\nEvaluation completed!")
    
    else:
        print("Please provide either --test-file for full evaluation or --text for single prediction")


if __name__ == '__main__':
    main()
