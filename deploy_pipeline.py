#!/usr/bin/env python3
"""
Deployment pipeline for publishing trained models to Hugging Face Hub.
"""

import os
import yaml
import logging
from pathlib import Path
from src.deployment.hub_client import save_model_for_huggingface, upload_to_huggingface
from src.training.classifier import BERTBinaryClassifier
from transformers import AutoTokenizer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deployment.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_deployment_config(config_path: str = "src/deployment/config.yaml") -> dict:
    """Load deployment configuration."""
    return load_config(config_path)

def find_latest_checkpoint(model_dir: str = "data/model") -> str:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory {model_dir} not found")
    
    checkpoints = list(model_path.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    
    best_checkpoints = [ckpt for ckpt in checkpoints if 'best' in ckpt.name.lower()]
    if best_checkpoints:
        latest_checkpoint = max(best_checkpoints, key=os.path.getmtime)
    else:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    
    return str(latest_checkpoint)

def deploy_model(
    checkpoint_path: str,
    repo_name: str,
    training_config_path: str = "src/training/config.yaml",
    deployment_config_path: str = "src/deployment/config.yaml",
    private: bool = False,
    save_only: bool = False
):
    """Deploy model to Hugging Face Hub."""
    logger = logging.getLogger(__name__)
    
    # Load configurations
    training_config = load_config(training_config_path)
    deployment_config = load_deployment_config(deployment_config_path)
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load model
    model = BERTBinaryClassifier.load_from_checkpoint(checkpoint_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(training_config['model']['name'])
    
    logger.info(f"Saving model for Hugging Face format...")
    
    # Save model in Hugging Face format
    save_model_for_huggingface(
        model=model,
        tokenizer=tokenizer,
        output_dir=deployment_config['deployment']['output_dir'],
        model_name=repo_name,
        config=training_config
    )
    
    logger.info(f"Model saved to: {deployment_config['deployment']['output_dir']}")
    
    if not save_only:
        logger.info(f"Uploading to Hugging Face Hub: {repo_name}")
        
        # Upload to Hugging Face Hub
        upload_to_huggingface(
            model_path=deployment_config['deployment']['output_dir'],
            repo_name=repo_name,
            private=private,
            commit_message=deployment_config['huggingface']['commit_message']
        )
        
        logger.info("Upload completed successfully!")
        logger.info(f"Model published to: https://huggingface.co/{repo_name}")
    else:
        logger.info("Model saved locally. Set save_only=False to upload to Hugging Face Hub.")

def main():
    """Main deployment function."""
    logger = setup_logging()
    
    # Load deployment config
    deployment_config = load_deployment_config()
    
    # Auto-find latest checkpoint
    checkpoint_path = find_latest_checkpoint()
    logger.info(f"Auto-detected latest checkpoint: {checkpoint_path}")
    
    # Get repository name from config
    repo_name = deployment_config['huggingface']['repo_name']
    if not repo_name:
        raise ValueError("Repository name must be specified in src/deployment/config.yaml")
    
    # Deploy the model
    deploy_model(
        checkpoint_path=checkpoint_path,
        repo_name=repo_name,
        private=deployment_config['huggingface']['private'],
        save_only=deployment_config['deployment']['save_only']
    )

if __name__ == "__main__":
    main()
