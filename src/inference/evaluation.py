#!/usr/bin/env python3
"""
Model evaluation module for binary text classification.
Handles comprehensive evaluation, visualization, and reporting.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

from .inference import ModelInference

class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Setup paths
        self.output_dir = Path(config['evaluation']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and data
        inference_config = config['inference']
        self.model = ModelInference(
            checkpoint_path=inference_config['model_path'],
            model_name=inference_config['model_name'],
            device=inference_config['device']
        )
        
        self.validation_data = self._load_validation_data(config['data'])
        self.logger.info(f"Loaded {len(self.validation_data)} validation samples")
    
    def _load_validation_data(self, data_config: Dict[str, str]) -> pd.DataFrame:
        """Load and clean validation data."""
        csv_path = data_config['validation_csv']
        df = pd.read_csv(csv_path)
        
        # Clean data
        original_length = len(df)
        df = df.dropna(subset=[data_config['text_column'], data_config['label_column']])
        df = df[df[data_config['text_column']].astype(str).str.strip() != '']
        df = df[df[data_config['text_column']].astype(str).str.len() >= 10]
        
        # Convert labels to int
        df[data_config['label_column']] = df[data_config['label_column']].astype(int)
        
        cleaned_length = len(df)
        self.logger.info(f"Data cleaning: {original_length} -> {cleaned_length} rows ({original_length - cleaned_length} removed)")
        
        return df
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        self.logger.info("Starting detailed evaluation...")
        
        # Initialize results
        predictions = []
        confidences = []
        probabilities = []
        inference_times = []
        
        # Row-by-row prediction
        progress_interval = self.config['evaluation']['performance']['log_progress_interval']
        start_time = time.time()
        
        for idx, row in self.validation_data.iterrows():
            text = row[self.config['data']['text_column']]
            
            # Measure inference time
            pred_class, confidence, probs, pred_time = self.model.predict_with_timing(text)
            
            predictions.append(pred_class)
            confidences.append(confidence)
            probabilities.append(probs)
            inference_times.append(pred_time)
            
            if (idx + 1) % progress_interval == 0:
                self.logger.info(f"Processed {idx + 1}/{len(self.validation_data)} samples")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        y_true = self.validation_data[self.config['data']['label_column']].values
        y_pred = np.array(predictions)
        y_proba = np.array(probabilities)[:, 1]  # Probability of class 1
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        # Performance metrics
        avg_inference_time = np.mean(inference_times)
        total_inference_time = total_time
        samples_per_second = len(self.validation_data) / total_time
        
        # Store results
        results = {
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            'performance': {
                'total_samples': len(self.validation_data),
                'total_time_seconds': total_inference_time,
                'avg_inference_time_ms': avg_inference_time * 1000,
                'samples_per_second': samples_per_second
            },
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist(),
                'confidences': confidences,
                'inference_times': inference_times
            },
            'model_info': self.model.get_model_info()
        }
        
        self.logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        self.logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        self.logger.info(f"Inference speed: {samples_per_second:.1f} samples/second")
        
        return results
    
    def generate_visualizations(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive visualizations."""
        if not any(self.config['evaluation']['visualizations'].values()):
            self.logger.info("Skipping visualizations (disabled in config)")
            return
            
        self.logger.info("Generating visualizations...")
        
        y_true = np.array(results['predictions']['y_true'])
        y_pred = np.array(results['predictions']['y_pred'])
        y_proba = np.array(results['predictions']['y_proba'])
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        viz_config = self.config['evaluation']['visualizations']
        
        # Create main performance plot
        if any([viz_config['plot_confusion_matrix'], viz_config['plot_roc_curve'], 
                viz_config['plot_precision_recall'], viz_config['plot_probability_distribution']]):
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Confusion Matrix
            if viz_config['plot_confusion_matrix']:
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                           xticklabels=['Normal', 'Garbled'], yticklabels=['Normal', 'Garbled'])
                axes[0,0].set_title('Confusion Matrix')
                axes[0,0].set_xlabel('Predicted')
                axes[0,0].set_ylabel('Actual')
            else:
                axes[0,0].axis('off')
            
            # ROC Curve
            if viz_config['plot_roc_curve']:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = results['metrics']['roc_auc']
                axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                              label=f'ROC curve (AUC = {roc_auc:.3f})')
                axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[0,1].set_xlim([0.0, 1.0])
                axes[0,1].set_ylim([0.0, 1.05])
                axes[0,1].set_xlabel('False Positive Rate')
                axes[0,1].set_ylabel('True Positive Rate')
                axes[0,1].set_title('ROC Curve')
                axes[0,1].legend(loc="lower right")
            else:
                axes[0,1].axis('off')
            
            # Precision-Recall Curve
            if viz_config['plot_precision_recall']:
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
                axes[1,0].plot(recall_vals, precision_vals, color='blue', lw=2)
                axes[1,0].set_xlabel('Recall')
                axes[1,0].set_ylabel('Precision')
                axes[1,0].set_title('Precision-Recall Curve')
                axes[1,0].grid(True)
            else:
                axes[1,0].axis('off')
            
            # Probability Distribution
            if viz_config['plot_probability_distribution']:
                normal_probs = y_proba[y_true == 0]
                garbled_probs = y_proba[y_true == 1]
                axes[1,1].hist(normal_probs, bins=50, alpha=0.7, label='Normal', color='green')
                axes[1,1].hist(garbled_probs, bins=50, alpha=0.7, label='Garbled', color='red')
                axes[1,1].set_xlabel('Predicted Probability (Class 1)')
                axes[1,1].set_ylabel('Frequency')
                axes[1,1].set_title('Probability Distribution by True Class')
                axes[1,1].legend()
                axes[1,1].grid(True)
            else:
                axes[1,1].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Threshold Analysis
        if viz_config['plot_threshold_analysis']:
            self._plot_threshold_analysis(y_true, y_proba)
        
        # Performance Analysis
        if viz_config['plot_performance_analysis']:
            self._plot_performance_analysis(results)
        
        self.logger.info("Visualizations saved successfully")
    
    def _plot_threshold_analysis(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        """Plot threshold analysis for different metrics."""
        metrics_config = self.config['evaluation']['metrics']
        threshold_min, threshold_max = metrics_config['threshold_range']
        threshold_steps = metrics_config['threshold_steps']
        
        thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            accuracies.append(accuracy_score(y_true, y_pred_thresh))
            precisions.append(precision_score(y_true, y_pred_thresh, zero_division=0))
            recalls.append(recall_score(y_true, y_pred_thresh, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))
        
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, accuracies, label='Accuracy', marker='o')
        plt.plot(thresholds, precisions, label='Precision', marker='s')
        plt.plot(thresholds, recalls, label='Recall', marker='^')
        plt.plot(thresholds, f1_scores, label='F1-Score', marker='d')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Default Threshold (0.5)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_analysis(self, results: Dict[str, Any]) -> None:
        """Plot inference performance analysis."""
        inference_times = results['predictions']['inference_times']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Inference time distribution
        axes[0].hist(inference_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Inference Time (seconds)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Inference Time Distribution')
        axes[0].grid(True)
        
        # Cumulative inference time
        cumulative_times = np.cumsum(inference_times)
        axes[1].plot(range(len(cumulative_times)), cumulative_times, color='green')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Cumulative Time (seconds)')
        axes[1].set_title('Cumulative Inference Time')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_markdown_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive markdown report."""
        if not self.config['evaluation']['reports']['generate_markdown']:
            self.logger.info("Skipping markdown report (disabled in config)")
            return
            
        self.logger.info("Generating markdown report...")
        
        metrics = results['metrics']
        performance = results['performance']
        model_info = results['model_info']
        
        report = f"""# Model Evaluation Report

## Overview
This report provides a comprehensive evaluation of the binary text classification model on the validation dataset.

## Dataset Information
- **Total Samples**: {performance['total_samples']:,}
- **Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- **Device**: {model_info['device']}
- **Total Parameters**: {model_info['total_parameters']:,}
- **Model Size**: {model_info['model_size_mb']:.1f} MB
- **Max Sequence Length**: {model_info['max_length']}

## Model Performance Metrics

### Classification Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | {metrics['accuracy']:.4f} |
| **Precision** | {metrics['precision']:.4f} |
| **Recall** | {metrics['recall']:.4f} |
| **F1-Score** | {metrics['f1_score']:.4f} |
| **ROC-AUC** | {metrics['roc_auc']:.4f} |

### Inference Performance
| Metric | Value |
|--------|-------|
| **Total Inference Time** | {performance['total_time_seconds']:.2f} seconds |
| **Average Inference Time** | {performance['avg_inference_time_ms']:.2f} ms |
| **Throughput** | {performance['samples_per_second']:.1f} samples/second |

## Detailed Analysis

### Classification Report
```
{classification_report(results['predictions']['y_true'], results['predictions']['y_pred'], target_names=['Normal', 'Garbled'])}
```

### Confusion Matrix Analysis
The confusion matrix shows the model's performance across both classes:
- **True Negatives (Normal correctly classified)**: {confusion_matrix(results['predictions']['y_true'], results['predictions']['y_pred'])[0,0]}
- **False Positives (Normal misclassified as Garbled)**: {confusion_matrix(results['predictions']['y_true'], results['predictions']['y_pred'])[0,1]}
- **False Negatives (Garbled misclassified as Normal)**: {confusion_matrix(results['predictions']['y_true'], results['predictions']['y_pred'])[1,0]}
- **True Positives (Garbled correctly classified)**: {confusion_matrix(results['predictions']['y_true'], results['predictions']['y_pred'])[1,1]}

## Visualizations

### 1. Model Performance Plots
![Evaluation Plots](evaluation_plots.png)

### 2. Threshold Analysis
![Threshold Analysis](threshold_analysis.png)

### 3. Performance Analysis
![Performance Analysis](performance_analysis.png)

## Recommendations

### Model Performance
- **ROC-AUC of {metrics['roc_auc']:.3f}** indicates {'excellent' if metrics['roc_auc'] > 0.9 else 'good' if metrics['roc_auc'] > 0.8 else 'fair'} discriminative ability
- **F1-Score of {metrics['f1_score']:.3f}** shows {'excellent' if metrics['f1_score'] > 0.9 else 'good' if metrics['f1_score'] > 0.8 else 'fair'} balance between precision and recall

### Inference Speed
- **{performance['samples_per_second']:.1f} samples/second** throughput is {'excellent' if performance['samples_per_second'] > 100 else 'good' if performance['samples_per_second'] > 50 else 'moderate'} for production deployment
- Average inference time of **{performance['avg_inference_time_ms']:.2f}ms** per sample

## Files Generated
- `evaluation_plots.png` - Main performance visualizations
- `threshold_analysis.png` - Threshold sensitivity analysis  
- `performance_analysis.png` - Inference performance metrics
- `detailed_results.json` - Raw evaluation data
- `detailed_predictions.csv` - Individual predictions and probabilities

---
*Report generated automatically by the detailed evaluation script*
"""
        
        with open(self.output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)
        
        self.logger.info("Markdown report saved successfully")
    
    def save_detailed_results(self, results: Dict[str, Any]) -> None:
        """Save detailed results and predictions."""
        reports_config = self.config['evaluation']['reports']
        
        self.logger.info("Saving detailed results...")
        
        # Save raw results as JSON
        if reports_config['generate_json']:
            with open(self.output_dir / 'detailed_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        # Save predictions as CSV
        if reports_config['generate_csv']:
            predictions_df = self.validation_data.copy()
            predictions_df['predicted_class'] = results['predictions']['y_pred']
            predictions_df['predicted_probability'] = results['predictions']['y_proba']
            predictions_df['confidence'] = results['predictions']['confidences']
            predictions_df['inference_time_ms'] = [t * 1000 for t in results['predictions']['inference_times']]
            predictions_df['correct'] = predictions_df[self.config['data']['label_column']] == predictions_df['predicted_class']
            
            predictions_df.to_csv(self.output_dir / 'detailed_predictions.csv', index=False)
        
        self.logger.info("Detailed results saved successfully")
