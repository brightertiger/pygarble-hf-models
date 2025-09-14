"""
PyTorch Lightning model for sentence transformer fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from torchmetrics import Accuracy, Precision, Recall, F1Score
from typing import Dict, Any, Optional
import numpy as np


class SentenceTransformerClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning sentence transformers for binary classification.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_classes: int = 2,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_length = max_length
        
        # Load the pre-trained transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize metrics
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")
        
        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.test_precision = Precision(task="binary")
        
        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")
        self.test_recall = Recall(task="binary")
        
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")
        
    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        
        # Pass through classification head
        logits = self.classifier(pooled_output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.train_f1(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Linear warmup scheduler
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def predict(self, texts: list, tokenizer, device: str = 'cpu') -> list:
        """
        Make predictions on a list of texts.
        
        Args:
            texts: List of input texts
            tokenizer: Tokenizer instance
            device: Device to run inference on
        
        Returns:
            List of predicted probabilities and labels
        """
        self.eval()
        self.to(device)
        
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Forward pass
                logits = self(input_ids, attention_mask)
                probs = F.softmax(logits, dim=1)
                pred_label = torch.argmax(probs, dim=1).item()
                pred_prob = probs[0][pred_label].item()
                
                predictions.append({
                    'text': text,
                    'predicted_label': pred_label,
                    'confidence': pred_prob,
                    'probabilities': probs[0].cpu().numpy().tolist()
                })
        
        return predictions
