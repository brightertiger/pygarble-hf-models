"""
Utilities for uploading models to Hugging Face Hub.
"""

import os
import torch
import json
from typing import Dict, Any, Optional
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoConfig
import shutil
from sklearn.metrics import confusion_matrix
import numpy as np


def load_evaluation_metrics(eval_dir: str = "data/evaluation") -> Dict[str, Any]:
    eval_path = Path(eval_dir) / "detailed_results.json"
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            results = json.load(f)
        
        metrics = results['metrics']
        performance = results['performance']
        
        y_true = np.array(results['predictions']['y_true'])
        y_pred = np.array(results['predictions']['y_pred'])
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'throughput': performance['samples_per_second'],
            'avg_inference_time': performance['avg_inference_time_ms'],
            'total_parameters': None,
            'model_size_mb': None,
            'validation_samples': performance['total_samples'],
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }
    return None


def clean_output_directory(output_dir: str):
    """Clean the output directory before saving new model files."""
    if os.path.exists(output_dir):
        print(f"Cleaning existing files in {output_dir}")
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"  Removed: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  Removed directory: {item}")
            except Exception as e:
                print(f"  Warning: Could not remove {item}: {e}")
        print("Directory cleaned successfully")
    else:
        print(f"Creating new directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)


def save_model_for_huggingface(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: str,
    model_name: str,
    config: Dict[str, Any]
):
    """
    Save model and tokenizer in Hugging Face format.
    
    Args:
        model: Trained PyTorch model
        tokenizer: Tokenizer instance
        output_dir: Directory to save the model
        model_name: Name of the model
        config: Configuration dictionary
    """
    clean_output_directory(output_dir)
    
    print("\n📦 Saving model files...")
    
    print("  Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    
    print("  Saving model with classification head...")
    hf_model = model.model
    hf_model.config.id2label = {0: "NORMAL", 1: "GARBLED"}
    hf_model.config.label2id = {"NORMAL": 0, "GARBLED": 1}
    
    hf_model.save_pretrained(output_dir)
    print("  ✓ Model saved in HuggingFace format")
    
    eval_metrics = load_evaluation_metrics()
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    if eval_metrics:
        eval_metrics['total_parameters'] = total_params
        eval_metrics['model_size_mb'] = model_size_mb
        performance_data = eval_metrics
        print(f"Loaded evaluation metrics from data/evaluation/detailed_results.json")
    else:
        performance_data = {
            "total_parameters": total_params,
            "model_size_mb": model_size_mb
        }
        print("Warning: No evaluation metrics found. Using model stats only.")
    
    create_model_card(output_dir, model_name, config, performance_data)
    
    metadata_config = {
        "model_type": "BertForSequenceClassification",
        "base_model": config['model']['name'],
        "max_length": config['model']['max_length'],
        "task": "text-classification",
        "pipeline_tag": "text-classification",
        "id2label": {
            "0": "NORMAL",
            "1": "GARBLED"
        },
        "label2id": {
            "NORMAL": 0,
            "GARBLED": 1
        },
        "performance": performance_data
    }
    
    print("  Saving metadata...")
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata_config, f, indent=2)
    
    create_inference_script(output_dir, model_name)
    
    print("\n✅ Model preparation complete!")
    print(f"\n📁 Files ready for deployment in '{output_dir}':")
    for file in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb > 1:
                print(f"  ✓ {file} ({size_mb:.1f} MB)")
            else:
                size_kb = os.path.getsize(file_path) / 1024
                print(f"  ✓ {file} ({size_kb:.1f} KB)")
    print()


def create_model_card(output_dir: str, model_name: str, config: Dict[str, Any], performance_data: Dict[str, Any]):
    """Copy the user-focused README.md to the model directory."""
    
    main_readme_path = "README.md"
    model_readme_path = os.path.join(output_dir, "README.md")
    
    if os.path.exists(main_readme_path):
        shutil.copy2(main_readme_path, model_readme_path)
        print(f"✓ Copied README.md to model directory")
    else:
        raise FileNotFoundError(f"Main README.md not found at {main_readme_path}. Please ensure the comprehensive README.md exists.")



def upload_to_huggingface(
    model_path: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload fine-tuned BERT binary classifier model"
):
    """
    Upload model to Hugging Face Hub using the modern HfApi.
    
    Args:
        model_path: Path to the saved model directory
        repo_name: Name of the repository on Hugging Face Hub
        private: Whether the repository should be private
        commit_message: Commit message for the upload
    """
    try:
        # Create repository if it doesn't exist
        create_repo(repo_name, private=private, exist_ok=True)
        print(f"Repository {repo_name} created/verified")
        
        # Use HfApi for modern upload approach
        api = HfApi()
        
        # Upload all files in the model directory
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            commit_message=commit_message,
            repo_type="model"
        )
        
        print(f"Model successfully uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"Error uploading to Hugging Face: {str(e)}")
        raise


def create_inference_script(output_dir: str, model_name: str):
    inference_script = '''#!/usr/bin/env python3
from transformers import pipeline
import sys

def main():
    print("Loading model...")
    classifier = pipeline("text-classification", model=".", device=-1)
    print("Model loaded successfully!\\n")
    
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        result = classifier(input_text)[0]
        print(f"Text: {input_text}")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['score']:.2%}")
    else:
        print("Garbled Text Detector - Testing with sample texts")
        print("=" * 70)
        
        sample_texts = [
            "This is a normal, well-formed text sample.",
            "The quick brown fox jumps over the lazy dog.",
            "H3ll0 w0rld! Th1s 1s g4rbl3d t3xt.",
            "Xkcd fjkdsl lorem ipsum dfkjsld dolor sit.",
            "I love this amazing product! It works great.",
            "asdfkj lksdjf weroi woeirj woeiruwo eiru",
            "Machine learning models are powerful tools."
        ]
        
        results = classifier(sample_texts)
        
        for text, result in zip(sample_texts, results):
            print(f"\\nText: {text[:60]}..." if len(text) > 60 else f"\\nText: {text}")
            print(f"  → {result['label']} (confidence: {result['score']:.2%})")
        
        print("\\n" + "=" * 70)
        print("\\nUsage: python inference.py [text to classify]")
        print("Example: python inference.py \\"Check if this text is garbled\\"")

if __name__ == "__main__":
    main()
'''
    
    with open(os.path.join(output_dir, "inference.py"), "w") as f:
        f.write(inference_script)
    print(f"Created simple inference script using pipeline API: {output_dir}/inference.py")
