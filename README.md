# Sentence Transformer Classifier

A production-ready library for fine-tuning sentence transformer models for binary text classification using PyTorch Lightning, with seamless Hugging Face Hub integration.

## Features

- 🚀 **Easy-to-use**: Simple CLI interface for training and evaluation
- ⚡ **PyTorch Lightning**: Leverages PyTorch Lightning for efficient training
- 🔧 **Configurable**: YAML-based configuration system
- 📊 **Comprehensive Evaluation**: Detailed metrics and visualizations
- 🤗 **Hugging Face Integration**: Easy model upload to Hugging Face Hub
- 📈 **Monitoring**: Optional Weights & Biases integration
- 🎯 **Binary Classification**: Optimized for binary text classification tasks

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pygarble-hf-models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Weights & Biases for experiment tracking:
```bash
pip install wandb
wandb login
```

## Quick Start

### 1. Create Sample Data

First, create sample data to test the pipeline:

```bash
./train --create-sample-data
```

This creates sample training, validation, and test data in the `data/` directory.

### 2. Train a Model

Train your model using the default configuration:

```bash
./train
```

Or customize the training with different configurations:

```bash
./train --config src/configs/production.yaml
./train --config src/configs/research.yaml
```

### 3. Evaluate the Model

Evaluate your trained model:

```bash
./evaluate --model checkpoints/best-model.ckpt --test-file data/test.csv
```

Or make predictions on single texts:

```bash
./evaluate --model checkpoints/best-model.ckpt --text "This is a great product!"
```

### 4. Publish to Hugging Face

Publish your trained model to Hugging Face Hub:

```bash
./publish --checkpoint checkpoints/best-model.ckpt --repo-name your-username/your-model-name
```

## Command Line Interface

The library provides three main CLI commands:

- **`./train`** - Train models with various configurations
- **`./evaluate`** - Evaluate models and make predictions  
- **`./publish`** - Publish models to Hugging Face Hub

All commands support the same argument format without requiring argparse:

```bash
# Training
./train [--config <config_file>] [--data-dir <data_directory>] [--create-sample-data] [--test-only] [--checkpoint <checkpoint>]

# Evaluation  
./evaluate --model <checkpoint_path> [--config <config_file>] [--test-file <test_file>] [--text <text>] [--output-dir <output_dir>]

# Publishing
./publish --checkpoint <checkpoint_path> [--config <config_file>] [--repo-name <repo_name>] [--private] [--save-only]
```

## Data Format

Your data should be in CSV format with the following columns:

- `text`: The input text to classify
- `label`: The binary label (0 or 1, or string labels that will be converted)

Example:
```csv
text,label
"This is amazing!",1
"I don't like this.",0
"Great quality product.",1
```

## Configuration

The library comes with three pre-configured setups:

### Default Configuration (`configs/default.yaml`)
- Balanced settings for general use
- Moderate batch size and learning rate
- Good for most binary classification tasks

### Production Configuration (`configs/production.yaml`)
- Optimized for production deployment
- Larger batch sizes and efficient settings
- Includes Weights & Biases logging

### Research Configuration (`configs/research.yaml`)
- Settings for research and experimentation
- Longer training with more epochs
- Higher precision for detailed analysis

You can also create custom configurations by copying and modifying any of these files.

## Usage Examples

### Training with Custom Data

```bash
# Train with your own data
./train --data-dir /path/to/data --config src/configs/production.yaml

# Train with Weights & Biases logging (enabled in production config)
./train --config src/configs/production.yaml
```

### Evaluation

```bash
# Full evaluation on test set
./evaluate --model checkpoints/best-model.ckpt --test-file data/test.csv

# Single text prediction
./evaluate --model checkpoints/best-model.ckpt --text "Your text here"

# Evaluation with custom output directory
./evaluate --model checkpoints/best-model.ckpt --test-file data/test.csv --output-dir results/
```

### Model Publishing

```bash
# Publish to Hugging Face Hub
./publish --checkpoint checkpoints/best-model.ckpt --repo-name username/model-name

# Publish as private repository
./publish --checkpoint checkpoints/best-model.ckpt --repo-name username/model-name --private

# Save model locally without publishing
./publish --checkpoint checkpoints/best-model.ckpt --save-only
```

## Project Structure

```
sentence-transformer-classifier/
├── src/
│   ├── core/
│   │   ├── classifier.py     # PyTorch Lightning model
│   │   └── dataset.py        # Data loading utilities
│   ├── training/
│   │   └── trainer.py        # Training pipeline
│   ├── evaluation/
│   │   └── benchmark.py      # Evaluation and metrics
│   ├── deployment/
│   │   ├── hub_client.py     # Hugging Face Hub integration
│   │   └── publisher.py      # Model publishing
│   └── configs/
│       ├── default.yaml      # Default configuration
│       ├── production.yaml   # Production settings
│       └── research.yaml      # Research settings
├── train                      # CLI: Training
├── evaluate                   # CLI: Evaluation
├── publish                    # CLI: Publishing
├── requirements.txt           # Dependencies
├── setup.py                  # Package setup
├── Makefile                  # Common operations
└── README.md                 # Documentation
```

## Model Architecture

The model consists of:

1. **Pre-trained Sentence Transformer**: Base model (e.g., `all-MiniLM-L6-v2`)
2. **Classification Head**: Two-layer MLP with dropout
3. **Binary Classification**: Outputs probabilities for two classes

## Training Features

- **Automatic Mixed Precision**: Faster training with reduced memory usage
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Linear warmup followed by constant learning rate
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best model based on validation loss
- **Comprehensive Logging**: Tracks all metrics during training

## Evaluation Metrics

The evaluation script provides:

- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class and macro-averaged metrics
- **Confusion Matrix**: Visual representation of predictions
- **Detailed Results**: Per-sample predictions with confidence scores

## Hugging Face Integration

The upload script:

- Converts the PyTorch Lightning model to Hugging Face format
- Creates a comprehensive model card
- Generates an inference script for easy usage
- Handles repository creation and file upload

## Advanced Usage

### Custom Model Architecture

You can modify the model architecture in `src/model.py`:

```python
class SentenceTransformerClassifier(pl.LightningModule):
    def __init__(self, ...):
        # Customize the classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),  # Custom architecture
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
```

### Custom Data Loading

Modify `src/data_utils.py` to handle different data formats:

```python
def load_data_from_json(file_path: str):
    # Custom JSON loading logic
    pass
```

### Custom Metrics

Add custom metrics in the model class:

```python
def __init__(self, ...):
    self.custom_metric = CustomMetric()
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Enable mixed precision training
3. **Poor Performance**: Try different learning rates or model architectures
4. **Upload Errors**: Ensure you're logged into Hugging Face Hub

### Performance Tips

- Use mixed precision training (`precision: "16-mixed"`)
- Increase batch size if you have more GPU memory
- Use gradient accumulation for effective larger batch sizes
- Enable Weights & Biases for experiment tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
