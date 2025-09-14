#!/usr/bin/env python3
"""
Training script for fine-tuning sentence transformers for binary classification.
"""

import yaml
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from src.core.classifier import SentenceTransformerClassifier
from src.core.dataset import (
    load_data_from_csv, 
    create_data_loaders, 
    split_data, 
    create_sample_data
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(config: dict, project_name: str = None):
    """Setup logger (Wandb or default)."""
    if config.get('logging', {}).get('use_wandb', False):
        logger = WandbLogger(
            project=project_name or config['logging']['project_name'],
            save_dir='logs'
        )
    else:
        logger = None
    return logger


def setup_callbacks(config: dict):
    """Setup training callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def main():
    # Check for command line arguments
    create_sample_data = '--create-sample-data' in sys.argv
    test_only = '--test-only' in sys.argv
    
    # Get config file from args or use default
    config_file = 'src/configs/default.yaml'
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            break
    
    # Get data directory from args or use default
    data_dir = 'data'
    for i, arg in enumerate(sys.argv):
        if arg == '--data-dir' and i + 1 < len(sys.argv):
            data_dir = sys.argv[i + 1]
            break
    
    # Get checkpoint from args
    checkpoint = None
    for i, arg in enumerate(sys.argv):
        if arg == '--checkpoint' and i + 1 < len(sys.argv):
            checkpoint = sys.argv[i + 1]
            break
    
    # Load configuration
    config = load_config(config_file)
    
    # Create sample data if requested
    if create_sample_data:
        print("Creating sample data...")
        create_sample_data(data_dir)
        print("Sample data created successfully!")
        return
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load data
    print("Loading data...")
    
    # Check if separate train/val files exist
    train_file = os.path.join(data_dir, config['data']['train_file'].split('/')[-1])
    val_file = os.path.join(data_dir, config['data']['val_file'].split('/')[-1])
    
    if os.path.exists(train_file) and os.path.exists(val_file):
        # Load separate train and validation files
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
    else:
        # Load single file and split
        data_file = os.path.join(data_dir, config['data']['train_file'].split('/')[-1])
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        texts, labels = load_data_from_csv(
            data_file, 
            config['data']['text_column'], 
            config['data']['label_column']
        )
        
        train_texts, train_labels, val_texts, val_labels = split_data(
            texts, labels, config['data']['val_split']
        )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_texts, train_labels, val_texts, val_labels,
        tokenizer,
        config['training']['batch_size'],
        config['model']['max_length'],
        config['hardware']['num_workers']
    )
    
    # Initialize model
    model = SentenceTransformerClassifier(
        model_name=config['model']['name'],
        num_classes=2,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        dropout=config['model']['dropout'],
        max_length=config['model']['max_length']
    )
    
    if test_only:
        # Test mode
        if not checkpoint:
            raise ValueError("Checkpoint path required for testing")
        
        print(f"Loading model from checkpoint: {checkpoint}")
        model = SentenceTransformerClassifier.load_from_checkpoint(checkpoint)
        
        # Load test data
        test_file = os.path.join(data_dir, config['data']['test_file'].split('/')[-1])
        if os.path.exists(test_file):
            test_texts, test_labels = load_data_from_csv(
                test_file, 
                config['data']['text_column'], 
                config['data']['label_column']
            )
            
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
            trainer.test(model, test_loader)
        else:
            print("No test file found, skipping testing")
        
        return
    
    # Setup logger and callbacks
    logger = setup_logger(config)
    callbacks = setup_callbacks(config)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['logging']['log_every_n_steps']
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")
    print(f"Best model saved at: {callbacks[0].best_model_path}")
    
    # Test the model
    test_file = os.path.join(data_dir, config['data']['test_file'].split('/')[-1])
    if os.path.exists(test_file):
        print("Running final evaluation on test set...")
        test_texts, test_labels = load_data_from_csv(
            test_file, 
            config['data']['text_column'], 
            config['data']['label_column']
        )
        
        _, test_loader = create_data_loaders(
            test_texts, test_labels, test_texts, test_labels,
            tokenizer,
            config['training']['batch_size'],
            config['model']['max_length'],
            config['hardware']['num_workers']
        )
        
        trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
