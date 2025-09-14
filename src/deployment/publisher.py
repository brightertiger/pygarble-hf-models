#!/usr/bin/env python3
"""
Script to upload fine-tuned models to Hugging Face Hub.
"""

import yaml
import os
import sys
import torch
from transformers import AutoTokenizer

from src.core.classifier import SentenceTransformerClassifier
from src.deployment.hub_client import save_model_for_huggingface, upload_to_huggingface


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Get checkpoint from args (required)
    checkpoint = None
    for i, arg in enumerate(sys.argv):
        if arg == '--checkpoint' and i + 1 < len(sys.argv):
            checkpoint = sys.argv[i + 1]
            break
    
    if not checkpoint:
        print("Error: --checkpoint argument is required")
        print("Usage: ./publish --checkpoint <checkpoint_path> [--config <config_file>] [--repo-name <repo_name>] [--private] [--save-only]")
        return
    
    # Get config file from args or use default
    config_file = 'src/configs/default.yaml'
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            break
    
    # Get repo name from args
    repo_name = None
    for i, arg in enumerate(sys.argv):
        if arg == '--repo-name' and i + 1 < len(sys.argv):
            repo_name = sys.argv[i + 1]
            break
    
    # Check for private flag
    private = '--private' in sys.argv
    
    # Get commit message from args or use default
    commit_message = 'Upload fine-tuned sentence transformer model'
    for i, arg in enumerate(sys.argv):
        if arg == '--commit-message' and i + 1 < len(sys.argv):
            commit_message = sys.argv[i + 1]
            break
    
    # Get output directory from args or use default
    output_dir = 'hf_model'
    for i, arg in enumerate(sys.argv):
        if arg == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            break
    
    # Check for save-only flag
    save_only = '--save-only' in sys.argv
    
    # Load configuration
    config = load_config(config_file)
    
    # Determine repository name
    repo_name = repo_name or config['huggingface']['repo_name']
    
    if not repo_name:
        raise ValueError("Repository name must be specified either in config or via --repo-name")
    
    print(f"Loading model from checkpoint: {checkpoint}")
    
    # Load model
    model = SentenceTransformerClassifier.load_from_checkpoint(checkpoint)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    print(f"Saving model for Hugging Face format...")
    
    # Save model in Hugging Face format
    save_model_for_huggingface(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        model_name=repo_name,
        config=config
    )
    
    print(f"Model saved to: {output_dir}")
    
    if not save_only:
        print(f"Uploading to Hugging Face Hub: {repo_name}")
        
        # Upload to Hugging Face Hub
        upload_to_huggingface(
            model_path=output_dir,
            repo_name=repo_name,
            private=private,
            commit_message=commit_message
        )
        
        print("Upload completed successfully!")
    else:
        print("Model saved locally. Remove --save-only flag to upload to Hugging Face Hub.")


if __name__ == '__main__':
    main()
