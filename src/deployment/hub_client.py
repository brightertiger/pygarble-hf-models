"""
Utilities for uploading models to Hugging Face Hub.
"""

import os
import torch
import json
from typing import Dict, Any, Optional
from huggingface_hub import HfApi, Repository, create_repo
from transformers import AutoTokenizer, AutoModel
import shutil


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
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Extract the transformer part of the model
    transformer_model = model.transformer
    
    # Save the transformer model
    transformer_model.save_pretrained(output_dir)
    
    # Save the classification head separately
    classifier_state = model.classifier.state_dict()
    torch.save(classifier_state, os.path.join(output_dir, "classifier_head.pt"))
    
    # Create model card
    create_model_card(output_dir, model_name, config)
    
    # Create config.json
    model_config = {
        "model_type": "sentence_transformer_classifier",
        "base_model": config['model']['name'],
        "num_classes": 2,
        "max_length": config['model']['max_length'],
        "dropout": config['model']['dropout'],
        "architecture": "transformer + classification_head"
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)


def create_model_card(output_dir: str, model_name: str, config: Dict[str, Any]):
    """Create a model card (README.md) for the Hugging Face model."""
    
    model_card_content = f"""---
language: en
license: apache-2.0
tags:
- sentence-transformers
- text-classification
- binary-classification
- pytorch
- pytorch-lightning
---

# {model_name}

A fine-tuned sentence transformer model for binary text classification.

## Model Description

This model is based on `{config['model']['name']}` and has been fine-tuned for binary text classification using PyTorch Lightning.

## Training Details

- **Base Model**: {config['model']['name']}
- **Learning Rate**: {config['training']['learning_rate']}
- **Batch Size**: {config['training']['batch_size']}
- **Epochs**: {config['training']['num_epochs']}
- **Max Length**: {config['model']['max_length']}
- **Dropout**: {config['model']['dropout']}

## Usage

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
base_model = AutoModel.from_pretrained("{model_name}")

# Load classification head
classifier_head = torch.load("classifier_head.pt")

# Create the complete model
class SentenceTransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = base_model
        self.classifier = nn.Sequential(
            nn.Dropout({config['model']['dropout']}),
            nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout({config['model']['dropout']}),
            nn.Linear(base_model.config.hidden_size // 2, 2)
        )
        self.classifier.load_state_dict(classifier_head)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# Initialize model
model = SentenceTransformerClassifier()

# Example inference
text = "This is a sample text for classification."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length={config['model']['max_length']})

with torch.no_grad():
    logits = model(**inputs)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

print(f"Predicted class: {{predicted_class}}")
print(f"Confidence: {{confidence:.4f}}")
```

## Model Performance

The model was trained on binary classification data and achieved the following performance metrics:

- **Accuracy**: [To be filled after training]
- **Precision**: [To be filled after training]
- **Recall**: [To be filled after training]
- **F1-Score**: [To be filled after training]

## Training Data

The model was trained on [describe your training data here].

## Limitations and Bias

This model may have limitations and biases inherent to the training data and base model. Please evaluate the model's performance on your specific use case before deployment.

## Citation

```bibtex
@misc{{{model_name.replace('-', '_').replace('/', '_')},
  title={{{model_name}}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/{model_name}}}}}
}}
```
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card_content)


def upload_to_huggingface(
    model_path: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload fine-tuned sentence transformer model"
):
    """
    Upload model to Hugging Face Hub.
    
    Args:
        model_path: Path to the saved model directory
        repo_name: Name of the repository on Hugging Face Hub
        private: Whether the repository should be private
        commit_message: Commit message for the upload
    """
    try:
        # Create repository if it doesn't exist
        create_repo(repo_name, private=private, exist_ok=True)
        
        # Initialize repository
        repo = Repository(model_path, clone_from=repo_name)
        
        # Add all files to git
        repo.git_add()
        
        # Commit and push
        repo.git_commit(commit_message)
        repo.git_push()
        
        print(f"Model successfully uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"Error uploading to Hugging Face: {str(e)}")
        raise


def create_inference_script(output_dir: str, model_name: str):
    """Create an inference script for easy model usage."""
    
    inference_script = f'''"""
Inference script for {model_name}
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class SentenceTransformerClassifier(nn.Module):
    def __init__(self, model_name="{model_name}"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.transformer.config.hidden_size, self.transformer.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer.config.hidden_size // 2, 2)
        )
        
        # Load classification head weights
        classifier_weights = torch.load("classifier_head.pt", map_location="cpu")
        self.classifier.load_state_dict(classifier_weights)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def predict(text, model, tokenizer, device="cpu"):
    """Make prediction on a single text."""
    model.eval()
    model.to(device)
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    inputs = {{k: v.to(device) for k, v in inputs.items()}}
    
    with torch.no_grad():
        logits = model(**inputs)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {{
        "text": text,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities[0].cpu().numpy().tolist()
    }}


if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("{model_name}")
    model = SentenceTransformerClassifier("{model_name}")
    
    # Example usage
    sample_texts = [
        "This is a great product!",
        "I don't like this at all.",
        "Amazing service and quality."
    ]
    
    for text in sample_texts:
        result = predict(text, model, tokenizer)
        print(f"Text: {{result['text']}}")
        print(f"Predicted class: {{result['predicted_class']}}")
        print(f"Confidence: {{result['confidence']:.4f}}")
        print("-" * 50)
'''
    
    with open(os.path.join(output_dir, "inference.py"), "w") as f:
        f.write(inference_script)
