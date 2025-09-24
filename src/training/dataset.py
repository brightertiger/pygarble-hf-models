import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
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


def load_data_from_csv(file_path: str, text_column: str, label_column: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(file_path)
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    return texts, labels


def split_data(texts: List[str], labels: List[int], val_split: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[int], List[str], List[int]]:
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_split, random_state=random_state, stratify=labels
    )
    return train_texts, train_labels, val_texts, val_labels


def create_data_loaders(
    train_texts: List[str], 
    train_labels: List[int],
    val_texts: List[str], 
    val_labels: List[int],
    tokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    
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


def create_sample_data(data_dir: str = "data", num_samples: int = 1000):
    import os
    import random
    
    sample_texts = [
        "This is a normal business document with clear formatting and readable text.",
        "The quarterly earnings report shows strong performance across all sectors.",
        "Meeting minutes from the board of directors session held on March 15th.",
        "Investment prospectus outlining the fund's strategy and risk assessment.",
        "Legal contract terms and conditions for software licensing agreement.",
        "Research paper abstract describing the methodology and key findings.",
        "User manual for the new software application with installation instructions.",
        "News article covering the latest developments in renewable energy technology.",
        "Email correspondence between project stakeholders discussing timeline updates.",
        "Financial analysis report with market trends and investment recommendations."
    ]
    
    garbled_texts = [
        "TRANSACTION HISTORY - AUGUST 2023 \u0001\u0002 \u0001\u0002\u0003 \u0010\u000f\u000e \u00a2\u00b3\u00c4\u00d5\u00e6\u00f7\u0000",
        "AGREEMENT NO. 2023-LE-50998\u0012ñéäÇñâàæÖàêêôê\u001aÐìçç\u001cÅâùûîöíðÿ Parties:",
        "Business Meeting Minutes: Date: Â®Ø7ûå10/09/2023. Attendees: J. Adams, S. King, R. Taylor.",
        "Binary Data Mimicking Text: ¸Ñö¼¶]\u001f. Agenda Item 3: New Marketing Strategy for Q4.",
        "Discuss potential collaborations with Å\u008d¢åâê©ëøøï LLC. Budget allocation proposed for Q4 is §ßè420,000.",
        "RNA Extraction and Quantitative Real-Time PCR (qRT-PCR) Total RNA was extracted from cellular lysates",
        "The Growth Opportunities Fund (the 'Fund') aims to achieve long-term capital appreciation by investing primarily",
        "Meeting minutes from the board of directors session held on March 15th with binary data corruption",
        "Legal contract terms and conditions for software licensing agreement with encoding issues",
        "Research paper abstract describing the methodology and key findings with data corruption"
    ]
    
    texts = []
    labels = []
    
    for i in range(num_samples):
        if i % 2 == 0:
            texts.append(random.choice(sample_texts))
            labels.append(0)
        else:
            texts.append(random.choice(garbled_texts))
            labels.append(1)
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, 'sample_data.csv'), index=False)
    
    print(f"Created sample dataset with {num_samples} samples in {data_dir}/sample_data.csv")
