"""
Data loading and preprocessing utilities for sentence transformer fine-tuning.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple
import os


class TextClassificationDataset(Dataset):
    """Dataset class for text classification tasks."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data_from_csv(
    file_path: str,
    text_column: str = 'text',
    label_column: str = 'label'
) -> Tuple[List[str], List[int]]:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        text_column: Name of the text column
        label_column: Name of the label column
    
    Returns:
        Tuple of (texts, labels)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in data")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in data")
    
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # Convert labels to integers if they're strings
    if isinstance(labels[0], str):
        unique_labels = list(set(labels))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        labels = [label_to_id[label] for label in labels]
        print(f"Label mapping: {label_to_id}")
    
    return texts, labels


def create_data_loaders(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def split_data(
    texts: List[str],
    labels: List[int],
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Split data into train and validation sets.
    
    Args:
        texts: List of texts
        labels: List of labels
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels)
    """
    from sklearn.model_selection import train_test_split
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_split, random_state=random_seed, stratify=labels
    )
    
    return train_texts, train_labels, val_texts, val_labels


def create_sample_data(output_dir: str = "data"):
    """
    Create sample data for testing the pipeline.
    
    Args:
        output_dir: Directory to save sample data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample positive and negative texts
    positive_texts = [
        "This is an amazing product! I love it.",
        "Great service, highly recommended.",
        "Excellent quality and fast delivery.",
        "Outstanding customer support.",
        "Perfect! Exactly what I was looking for.",
        "Fantastic experience, will buy again.",
        "Top-notch quality, worth every penny.",
        "Brilliant solution to my problem.",
        "Outstanding performance and reliability.",
        "Exceptional value for money."
    ]
    
    negative_texts = [
        "Terrible product, complete waste of money.",
        "Poor quality and slow delivery.",
        "Worst customer service ever.",
        "Disappointed with this purchase.",
        "Not worth the price at all.",
        "Broken upon arrival, very frustrating.",
        "Awful experience, would not recommend.",
        "Subpar quality, expected much better.",
        "Waste of time and money.",
        "Completely unsatisfied with this."
    ]
    
    # Create labels (0 for negative, 1 for positive)
    positive_labels = [1] * len(positive_texts)
    negative_labels = [0] * len(negative_texts)
    
    # Combine data
    all_texts = positive_texts + negative_texts
    all_labels = positive_labels + negative_labels
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': all_texts,
        'label': all_labels
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/val/test
    train_df = df.iloc[:12]  # 12 samples
    val_df = df.iloc[12:16]  # 4 samples
    test_df = df.iloc[16:]   # 4 samples
    
    # Save to CSV files
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Sample data created in {output_dir}/")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
