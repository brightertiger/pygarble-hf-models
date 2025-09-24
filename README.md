# Garbled Text Detector

A state-of-the-art binary text classification model for detecting garbled, corrupted, or malformed text using fine-tuned BERT architecture.

## 🎯 Model Overview

This model is specifically designed to distinguish between normal, well-formed text and garbled/corrupted text. It achieves **96.1% accuracy** with excellent performance across all metrics, making it ideal for data quality assessment, text preprocessing pipelines, and content filtering applications.

### Key Features
- **High Accuracy**: 96.1% accuracy on validation set
- **Fast Inference**: 50.3 samples/second throughput (19.6ms per sample)
- **Robust Performance**: ROC-AUC of 0.992 indicating excellent discriminative ability
- **CPU Optimized**: Quantized for efficient CPU inference
- **Production Ready**: Comprehensive evaluation and deployment pipeline

## 🧠 Model Logic & Data Generation

### Core Problem & Solution
**Problem**: PDF parsing tools often produce corrupted output when encountering non-text elements, encoding issues, or malformed documents. This creates noise in downstream NLP pipelines and data processing workflows.

**Solution**: A binary classifier that can reliably distinguish between:
- **Normal text**: Clean, well-formatted text from successful PDF parsing
- **Garbled text**: Corrupted output from parsing failures, encoding issues, or binary data interpretation

### Data Generation Strategy
Our approach uses **synthetic data generation** with Large Language Models to create realistic PDF parsing scenarios:

#### **1. Normal Parsing Scenario (Label 0)**
- **Clean PDF Extraction**: Simulates successful text extraction from well-formatted PDFs
- **Domain Coverage**: Financial reports, legal documents, scientific articles, general business documents
- **Content Types**: Structured paragraphs, proper formatting, standard punctuation
- **Examples**: Annual reports, legal contracts, research papers, employee handbooks

#### **2. Gibberish Scenario (Label 1)**
- **PDF Parsing Failures**: Simulates corrupted output from parsing tools
- **Corruption Types**:
  - Binary data interpretation: `PDF-1.4%âãÏÓ1 0 obj<< /Type /Catalog`
  - Encoding corruption: `$1,250,000%âãÏÓRevenue%âãÏÓ$890,000`
  - Non-ASCII characters mixed with readable text
  - Mixed character sets and encoding failures
- **Realistic Failures**: Based on actual PDF parsing tool behaviors

#### **3. Domain-Specific Scenario (Label 0 - False Positive Prevention)**
- **Technical Content**: Legitimate content that might appear "garbled" to naive classifiers
- **Content Types**:
  - Mathematical symbols: ∑, ∫, ±, ≤, ≥, ≠, ∞, α, β, γ
  - Currency symbols: $, €, £, ¥, ₣, ₹, ₽, ₩
  - Technical notation: Scientific formulas, chemical equations (H₂SO₄)
  - Legal citations: U.S.C. references, C.F.R. regulations
  - Well-formatted tables and structured data
- **Purpose**: Prevents false positives on legitimate technical content

### Training Data Composition
- **Normal Parsing**: 33% - Clean PDF extraction output
- **Gibberish Detection**: 33% - Corrupted parsing failures
- **Domain-Specific**: 33% - Technical content (labeled as normal to prevent false positives)

### Model Architecture Logic
- **Base Model**: TinyBERT for efficiency and CPU optimization
- **Classification Head**: Simple linear layer on pooled BERT output
- **Quantization**: Dynamic quantization for production deployment
- **Sequence Length**: 512 tokens to handle typical document pages

### Key Innovation
The critical insight is **false positive prevention**: Technical documents with mathematical symbols, legal citations, and specialized notation should be classified as "normal" even though they contain non-standard characters. This prevents the model from flagging legitimate technical content as corrupted.

## 📊 Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | 96.1% | Overall classification accuracy |
| **Precision** | 92.0% | Precision for garbled text detection |
| **Recall** | 95.8% | Recall for garbled text detection |
| **F1-Score** | 93.9% | Harmonic mean of precision and recall |
| **ROC-AUC** | 99.2% | Area under ROC curve |

### Detailed Classification Report
```
              precision    recall  f1-score   support

      Normal       0.98      0.96      0.97      4588
     Garbled       0.92      0.96      0.94      2058

    accuracy                           0.96      6646
   macro avg       0.95      0.96      0.96      6646
weighted avg       0.96      0.96      0.96      6646
```

## 🏗️ Architecture & Approach

### Model Architecture
- **Base Model**: TinyBERT (huawei-noah/TinyBERT_General_4L_312D)
- **Architecture**: BERT + Classification Head
- **Parameters**: 14.4M parameters (54.7 MB)
- **Sequence Length**: 512 tokens
- **Quantization**: Dynamic quantization for CPU optimization

### Technical Approach

#### 1. Data Generation Strategy
Our approach leverages **PDF parsing simulation** using Large Language Models to create realistic text extraction scenarios:

**PDF Parsing Simulation Framework:**
- **Normal Parsing Scenario**: Simulates successful PDF text extraction with clean, readable output
- **Gibberish Scenario**: Simulates PDF parsing failures with non-ASCII characters, binary data, and encoding corruption
- **Domain-Specific Scenario**: Simulates legitimate technical content with symbols, equations, and specialized terminology (labeled as NORMAL to prevent false positives)

**Labeling Strategy:**
- **Label 0 (Normal)**: Clean PDF extraction + Domain-specific technical content
- **Label 1 (Garbled)**: Corrupted PDF parsing output with binary data and encoding failures

**Specific Corruption Types (Label 1 - Garbled):**
- **Binary Data Interpretation**: PDF internal structure interpreted as text
- **Encoding Corruption**: UTF-8 and character encoding failures  
- **Non-ASCII Characters**: Special symbols mixed with readable text
- **Mixed Character Sets**: Multiple encodings causing character substitution

**False Positive Prevention (Label 0 - Normal):**
- **Technical Symbols**: Mathematical operators, scientific notation, currency symbols
- **Table Structure**: Well-formatted tables with proper alignment
- **Domain Terminology**: Specialized jargon and technical vocabulary
- **Formatted Content**: Structured data, citations, and technical specifications

#### 2. Model Selection Rationale
**TinyBERT Choice:**
- **Efficiency**: 4-layer architecture vs 12-layer BERT-base
- **Speed**: 3x faster inference than BERT-base
- **Size**: 54.7MB vs 440MB for BERT-base
- **Performance**: Maintains 96%+ of BERT-base performance
- **CPU Optimization**: Ideal for production deployment

#### 3. Training Methodology

**PyTorch Lightning Framework:**
```python
# Key training components
- Dynamic quantization during training
- Early stopping with patience=3
- Gradient clipping (max_norm=1.0)
- Linear warmup scheduler (100 steps)
- AdamW optimizer with weight decay
```

**Quantization Strategy:**
- **Training**: Dynamic quantization for memory efficiency
- **Inference**: Quantized weights for CPU optimization
- **Impact**: 75% memory reduction, 2-3x speed improvement

**Data Preprocessing Pipeline:**
```python
# Robust data cleaning
1. NaN value detection and removal
2. Blank text filtering (< 10 characters)
3. Label validation and conversion
4. Tokenization with 512 max length
5. Dynamic padding for batch processing
```

#### 4. Loss Function & Optimization

**Cross-Entropy Loss:**
- Binary classification with balanced classes
- No class weighting (natural distribution)
- Stable convergence across epochs

**Optimization Details:**
- **Learning Rate**: 2e-5 (BERT-standard)
- **Batch Size**: 16 (memory-optimized)
- **Weight Decay**: 0.01 (L2 regularization)
- **Warmup**: Linear warmup over 100 steps
- **Gradient Clipping**: 1.0 (prevents exploding gradients)

#### 5. Evaluation Framework

**Comprehensive Metrics:**
- **Primary**: Accuracy, F1-Score, ROC-AUC
- **Class-Specific**: Precision/Recall per class
- **Threshold Analysis**: Optimal decision boundary
- **Speed Benchmarking**: Inference time per sample

**Validation Strategy:**
- **Holdout Set**: 20% of generated data
- **Cross-Validation**: Stratified sampling
- **Robustness Testing**: Edge cases and adversarial examples

#### 6. Production Optimization

**CPU Inference Pipeline:**
```python
# Quantization for production
model.bert = torch.quantization.quantize_dynamic(
    model.bert, {nn.Linear}, dtype=torch.qint8
)
model.classifier = torch.quantization.quantize_dynamic(
    model.classifier, {nn.Linear}, dtype=torch.qint8
)
```

**Memory Management:**
- **Model Size**: 54.7MB (vs 440MB unquantized)
- **RAM Usage**: <100MB during inference
- **Throughput**: 50.3 samples/second on CPU

### Data Scenarios & Corruption Types

#### Normal Text Categories:
- **News Articles**: Well-formed journalistic text
- **Technical Documentation**: Structured technical content
- **Conversational**: Natural dialogue and social media text
- **Academic**: Formal academic writing and research text

#### PDF Parsing Corruption Patterns:

**1. Binary Data Interpretation:**
```
PDF Structure: "PDF-1.4%âãÏÓ1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj"
Parsed as: PDF internal structure mixed with content
```

**2. Encoding Corruption:**
```
Original: "Revenue $1,250,000"
Corrupted: "$1,250,000%âãÏÓRevenue%âãÏÓ$890,000"
```

**3. Technical Symbol Mixing:**
```
Domain-Specific: "Currency Exchange Rates USD$1.00 EUR€0.85 GBP£0.73"
Mathematical: "ROI = (Gain - Cost) / Cost × 100%"
```

**4. Table Structure Corruption:**
```
Legal Citations: "Smith v. Jones (2023) High Supreme Court Brown v. White (2022) Medium"
Mixed Content: "Case Citation Precedent Weight Binding Authority"
```

#### Data Composition & Labeling:

**Label 0 (Normal) - Two Types:**
1. **Clean PDF Extraction**: Well-formatted text from financial, legal, scientific, and general documents
2. **Domain-Specific Technical Content**: Legitimate content with symbols, tables, and equations (prevents false positives)

**Label 1 (Garbled) - One Type:**
1. **PDF Parsing Failures**: Corrupted output with binary data, encoding issues, and mixed character sets

**Domain-Specific Content (Label 0 - Normal):**
- **Financial Documents**: Currency exchange tables, ROI formulas, portfolio analysis matrices
- **Legal Documents**: Case citation hierarchies, compliance matrices, legal precedent algorithms  
- **Scientific Articles**: Chemical reaction data, statistical formulas, research methodology tables
- **General Documents**: Technical specifications, system architecture, performance metrics

**Key Technical Elements (All Labeled as Normal):**
- **Mathematical Symbols**: ∑, ∫, ±, ≤, ≥, ≠, ∞, α, β, γ
- **Technical Notation**: Scientific notation, chemical formulas (H₂SO₄), algorithm pseudocode
- **Currency Symbols**: $, €, £, ¥, ₣, ₹, ₽, ₩
- **Legal Citations**: U.S.C. references, C.F.R. regulations, court case formats
- **Table Structure**: Well-formatted tables with proper alignment and structure

## 🚀 Quick Start

### Installation
```bash
pip install torch transformers huggingface-hub
```

### Setup (Required for Deployment)
To deploy models to Hugging Face Hub, you need to authenticate:

```bash
# Install huggingface_hub CLI
pip install huggingface_hub

# Login to Hugging Face (requires API token)
huggingface-cli login
```

**Getting your Hugging Face API Token:**
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New Token" 
3. Choose "Write" permissions
4. Copy the token and use it when prompted by `huggingface-cli login`

### Deployment
Once authenticated, you can deploy your trained model:

```bash
# Deploy to Hugging Face Hub
python deploy_pipeline.py
```

**Prerequisites for Deployment:**
- ✅ Hugging Face account with API token
- ✅ Trained model checkpoint in `data/model/`
- ✅ Updated repository name in `src/deployment/config.yaml`

### Basic Usage
```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import json

# Load tokenizer and base BERT model
tokenizer = AutoTokenizer.from_pretrained("brightertiger/garbled-text-detector")
base_model = AutoModel.from_pretrained("brightertiger/garbled-text-detector")

# Load classification head and dropout config
classifier_head = torch.load("classifier_head.pt", map_location="cpu")
with open("dropout_config.json", "r") as f:
    dropout_config = json.load(f)

# Create the complete model
class BERTBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = base_model
        self.dropout_layer = nn.Dropout(dropout_config["dropout_rate"])
        self.classifier = nn.Linear(base_model.config.hidden_size, 2)
        self.classifier.load_state_dict(classifier_head)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout_layer(pooled_output)
        logits = self.classifier(output)
        return logits

# Initialize model
model = BERTBinaryClassifier()
model.eval()

# Simple prediction function
def predict_text(text, model, tokenizer):
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    # Get prediction
    with torch.no_grad():
        logits = model(**inputs)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "text": text,
        "predicted_class": predicted_class,
        "confidence": confidence
    }

# Test examples
sample_texts = [
    "This is a normal, well-formed text sample.",
    "H3ll0 w0rld! Th1s 1s g4rbl3d t3xt.",
    "I love this amazing product!",
    "Currency Exchange Rates USD$1.00 EUR€0.85 GBP£0.73"
]

for text in sample_texts:
    result = predict_text(text, model, tokenizer)
    class_label = "GARBLED" if result['predicted_class'] == 1 else "NORMAL"
    print(f"Text: {result['text']}")
    print(f"Predicted: {class_label} (confidence: {result['confidence']:.3f})")
    print("-" * 50)
```

## 📈 Model Performance Analysis

### Confusion Matrix Results
- **True Negatives**: 4,417 (Normal correctly classified)
- **False Positives**: 171 (Normal misclassified as Garbled)
- **False Negatives**: 86 (Garbled misclassified as Normal)
- **True Positives**: 1,972 (Garbled correctly classified)

### Inference Performance
- **Throughput**: 50.3 samples/second
- **Average Latency**: 19.6ms per sample
- **Total Evaluation Time**: 132.22 seconds for 6,646 samples
- **Device**: Optimized for CPU inference with MPS support

## 🎛️ Configuration

### Model Parameters
```yaml
model:
  name: "huawei-noah/TinyBERT_General_4L_312D"
  max_length: 512
  dropout: 0.1
  quantization: true
  quantized_inference: true
```

### Training Configuration
```yaml
training:
  num_epochs: 1
  batch_size: 16
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 100
```

## 🔧 Use Cases

### Primary Applications
1. **Data Quality Assessment**: Filter corrupted text from datasets
2. **Content Moderation**: Identify malformed user-generated content
3. **Text Preprocessing**: Clean text before further NLP processing
4. **OCR Quality Control**: Validate OCR output quality
5. **Encoding Detection**: Identify text encoding issues

### Integration Examples
```python
# Batch processing for data cleaning
def clean_dataset(texts):
    clean_texts = []
    for text in texts:
        result = predict_text(text, model, tokenizer)
        if result['predicted_class'] == 0:  # Normal text
            clean_texts.append(text)
    return clean_texts

# Quality control pipeline
def quality_control_pipeline(text):
    result = predict_text(text, model, tokenizer)
    if result['predicted_class'] == 1 and result['confidence'] > 0.8:
        return {"status": "flagged", "reason": "garbled_text", "confidence": result['confidence']}
    return {"status": "approved", "confidence": result['confidence']}
```

## 📁 Repository Structure

```
├── README.md                    # This file
├── config.json                  # Model configuration
├── pytorch_model.bin            # BERT model weights
├── classifier_head.pt           # Classification head weights
├── dropout_config.json          # Dropout configuration
├── tokenizer.json              # Tokenizer files
├── tokenizer_config.json       # Tokenizer configuration
├── special_tokens_map.json     # Special tokens mapping
├── vocab.txt                   # Vocabulary file
└── inference.py                # Ready-to-use inference script
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Reporting issues
- Suggesting enhancements
- Submitting pull requests
- Adding new corruption scenarios

## 📜 License

This model is released under the Apache 2.0 License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library and model hosting
- **Huawei Noah's Ark Lab** for the TinyBERT base model
- **PyTorch Lightning** for the training framework
- **Open source community** for various tools and libraries

## 📊 Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{garbled_text_detector,
  title={Garbled Text Detector: A BERT-based Binary Classifier for Text Quality Assessment},
  author={BrighterTiger},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/brightertiger/garbled-text-detector}}
}
```

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact: [Your Contact Information]
- Documentation: [Link to detailed docs]

---

**Model Performance**: 96.1% accuracy, 99.2% ROC-AUC, 50.3 samples/sec throughput
**Last Updated**: September 2025
**Version**: 1.0.0
