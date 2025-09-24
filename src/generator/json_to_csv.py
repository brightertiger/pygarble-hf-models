#!/usr/bin/env python3

import json
import logging
import os
import sys
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_json_data(file_path: str) -> List[dict]:
    """Load and validate JSON data from checkpoint file."""
    logger.info(f"Loading data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = data.get('examples', [])
    logger.info(f"Found {len(examples)} examples in {file_path}")
    
    return examples


def extract_texts_and_labels(files_with_labels: List[Tuple[str, int]]) -> Tuple[List[str], List[int]]:
    """Extract texts and labels from multiple JSON files."""
    all_texts = []
    all_labels = []
    
    for file_path, label in files_with_labels:
        logger.info(f"Processing {file_path} with label {label}")
        
        examples = load_json_data(file_path)
        valid_count = 0
        
        for example in examples:
            text = example.get('page_text', '')
            if isinstance(text, str) and text.strip():
                all_texts.append(text.strip())
                all_labels.append(label)
                valid_count += 1
        
        logger.info(f"Extracted {valid_count} valid texts from {file_path}")
    
    logger.info(f"Total extracted: {len(all_texts)} texts")
    return all_texts, all_labels


def create_dataframe(texts: List[str], labels: List[int]) -> pd.DataFrame:
    """Create pandas DataFrame from texts and labels."""
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    logger.info(f"Created DataFrame with shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    
    return df


def split_and_save_data(df: pd.DataFrame, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """Split data into train/validation sets and save as CSV files."""
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label']
    )
    
    logger.info(f"Train set: {train_df.shape[0]} samples")
    logger.info(f"Validation set: {val_df.shape[0]} samples")
    
    logger.info("Train set label distribution:")
    logger.info(f"\n{train_df['label'].value_counts().sort_index()}")
    
    logger.info("Validation set label distribution:")
    logger.info(f"\n{val_df['label'].value_counts().sort_index()}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'validation.csv')
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    
    logger.info(f"Saved train set to: {train_path}")
    logger.info(f"Saved validation set to: {val_path}")


def main():
    files_with_labels = [
        ('data/data/domain_specific_content_scenario_checkpoint.json', 0),
        ('data/data/normal_parsing_scenario_checkpoint.json', 0),
        ('data/data/gibberish_scenario_checkpoint.json', 1),
    ]
    
    output_dir = 'data/data'
    test_size = 0.2
    random_state = 42
    
    try:
        logger.info("Starting data conversion process...")
        logger.info(f"Files to process: {[f[0] for f in files_with_labels]}")
        logger.info(f"Labels: domain_specific=0, normal_parsing=0, gibberish=1")
        
        texts, labels = extract_texts_and_labels(files_with_labels)
        df = create_dataframe(texts, labels)
        
        split_and_save_data(df, output_dir, test_size, random_state)
        
        logger.info("Data conversion completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


