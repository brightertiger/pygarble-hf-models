#!/usr/bin/env python3
"""
Simple training pipeline for binary text classification.
"""

import os
import logging
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer

from src.training.utils import load_config, get_device
from src.training.dataset import create_sample_data, load_data_from_csv, create_data_loaders
from src.training.classifier import BERTBinaryClassifier
from src.training.trainer import setup_callbacks, setup_logger

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_pipeline.log')
        ]
    )
    return logging.getLogger(__name__)

def auto_detect_hardware(config):
    """Auto-detect and configure hardware settings."""
    logger = logging.getLogger(__name__)
    
    # Auto-detect GPU availability
    if config['hardware'].get('auto_detect_gpu', False):
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            config['hardware']['devices'] = gpu_count
            logger.info(f"🚀 GPU detected: {gpu_count}x {device_name}")
            logger.info(f"Using GPU acceleration with {gpu_count} device(s)")
        elif torch.backends.mps.is_available():
            config['hardware']['devices'] = 1
            logger.info("🚀 Apple Silicon GPU (MPS) detected")
            logger.info("Using Apple Silicon GPU acceleration")
        else:
            config['hardware']['devices'] = "cpu"
            logger.info("💻 No GPU detected, using CPU")
            logger.info("Consider using GPU for faster training")
    
    # Adjust precision based on device
    if config['hardware']['devices'] == "cpu":
        config['hardware']['precision'] = "32"
        logger.info("Using 32-bit precision for CPU training")
    else:
        logger.info(f"Using {config['hardware']['precision']} precision for GPU training")
    
    # Adjust batch size based on available memory
    if config['hardware']['devices'] != "cpu":
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
                if gpu_memory < 8:
                    config['training']['batch_size'] = min(config['training']['batch_size'], 8)
                    logger.info(f"Reduced batch size to {config['training']['batch_size']} for {gpu_memory:.1f}GB GPU")
                elif gpu_memory >= 16:
                    config['training']['batch_size'] = min(config['training']['batch_size'] * 2, 32)
                    logger.info(f"Increased batch size to {config['training']['batch_size']} for {gpu_memory:.1f}GB GPU")
        except Exception as e:
            logger.warning(f"Could not detect GPU memory: {e}")
    
    return config

def clean_model_directory(model_dir):
    """Clean model directory before training."""
    logger = logging.getLogger(__name__)
    
    if os.path.exists(model_dir):
        checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
        if checkpoint_files:
            logger.info(f"Cleaning {len(checkpoint_files)} old checkpoint(s) from {model_dir}")
            for ckpt_file in checkpoint_files:
                ckpt_path = os.path.join(model_dir, ckpt_file)
                os.remove(ckpt_path)
                logger.info(f"  Removed: {ckpt_file}")
            logger.info("Model directory cleaned successfully")
    else:
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Created model directory: {model_dir}")

def train_model(config):
    """Train the model with given configuration."""
    logger = logging.getLogger(__name__)
    
    model_dir = config['callbacks']['model_checkpoint']['dirpath']
    clean_model_directory(model_dir)
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load data
    data_dir = config['data']['data_dir']
    train_file = os.path.join(data_dir, config['data']['train_file'])
    val_file = os.path.join(data_dir, config['data']['val_file'])
    
    logger.info(f"Loading data from {data_dir}")
    
    # Load train and validation data
    train_texts, train_labels = load_data_from_csv(
        train_file, 
        config['data']['text_column'], 
        config['data']['label_column']
    )
    
    val_texts, val_labels = load_data_from_csv(
        val_file, 
        config['data']['text_column'], 
        config['data']['label_column']
    )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_texts, train_labels, val_texts, val_labels,
        tokenizer,
        config['training']['batch_size'],
        config['model']['max_length'],
        config['hardware']['num_workers']
    )
    
    # Initialize model
    model = BERTBinaryClassifier(
        model_name=config['model']['name'],
        num_classes=2,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        dropout=config['model']['dropout'],
        max_length=config['model']['max_length'],
        quantization=config['model'].get('quantization', False),
        quantized_inference=config['model'].get('quantized_inference', False)
    )
    
    # Setup logger and callbacks
    logger_obj = setup_logger(config)
    callbacks = setup_callbacks(config)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        logger=logger_obj,
        callbacks=callbacks,
        log_every_n_steps=config['logging']['log_every_n_steps']
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    logger.info("Training completed!")
    return callbacks[0].best_model_path

def evaluate_model(config, checkpoint_path):
    """Evaluate the model with given checkpoint."""
    logger = logging.getLogger(__name__)
    
    # Load model from checkpoint
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = BERTBinaryClassifier.load_from_checkpoint(checkpoint_path)
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load test data
    data_dir = config['data']['data_dir']
    test_file = os.path.join(data_dir, config['data']['test_file'])
    val_file = os.path.join(data_dir, config['data']['val_file'])
    
    # Use test data if available, otherwise validation data
    eval_file = test_file if os.path.exists(test_file) else val_file
    logger.info(f"Using evaluation data: {eval_file}")
    
    test_texts, test_labels = load_data_from_csv(
        eval_file, 
        config['data']['text_column'], 
        config['data']['label_column']
    )
    
    # Create test data loader
    _, test_loader = create_data_loaders(
        test_texts, test_labels, test_texts, test_labels,
        tokenizer,
        config['training']['batch_size'],
        config['model']['max_length'],
        config['hardware']['num_workers']
    )
    
    # Setup trainer for testing
    trainer = pl.Trainer(
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        logger=False
    )
    
    # Test the model
    logger.info("Running evaluation...")
    results = trainer.test(model, test_loader)
    logger.info("Evaluation completed!")
    
    return results

def main():
    """Main training pipeline."""
    logger = setup_logging()
    
    # Load configuration
    config_path = "src/training/config.yaml"
    config = load_config(config_path)
    
    logger.info("Starting binary text classification training pipeline")
    
    # Auto-detect and configure hardware
    config = auto_detect_hardware(config)
    
    # Get data directory from config
    data_dir = config['data']['data_dir']
    
    # Check if sample data creation is needed
    train_file = os.path.join(data_dir, config['data']['train_file'])
    val_file = os.path.join(data_dir, config['data']['val_file'])
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        logger.info("Creating sample data for training")
        create_sample_data(data_dir)
    
    try:
        # Train the model
        best_checkpoint = train_model(config)
        logger.info(f"Best model saved at: {best_checkpoint}")
        
        # Evaluate the model
        results = evaluate_model(config, best_checkpoint)
        logger.info(f"Evaluation results: {results}")
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
