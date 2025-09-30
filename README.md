---
language:
- en
license: apache-2.0
tags:
- text-classification
- binary-classification
- garbled-text-detection
- data-quality
- pdf-parsing
- bert
- pytorch
library_name: transformers
pipeline_tag: text-classification
datasets: []
metrics:
- accuracy
- precision
- recall
- f1
- roc_auc
model-index:
- name: garbled-text-detector
  results:
  - task:
      type: text-classification
      name: Text Classification
    metrics:
    - type: accuracy
      value: 0.962
      name: Accuracy
    - type: f1
      value: 0.939
      name: F1 Score
    - type: precision
      value: 0.925
      name: Precision
    - type: recall
      value: 0.954
      name: Recall
    - type: roc_auc
      value: 0.991
      name: ROC-AUC
---

# Garbled Text Detector

Detect garbled, corrupted, or malformed text with **96.2% accuracy**. Perfect for data quality assessment, PDF parsing validation, and text preprocessing pipelines.

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Basic Usage

The simplest way to use the model:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="brightertiger/garbled-text-detector")

result = classifier("Your text here")[0]
print(f"Label: {result['label']}, Confidence: {result['score']:.2%}")
```

### Batch Processing

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="brightertiger/garbled-text-detector")

texts = [
    "This is normal, well-formed text.",
    "H3ll0 w0rld! Th1s 1s g4rbl3d t3xt.",
    "The quick brown fox jumps over the lazy dog."
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"{text[:50]:50} -> {result['label']} ({result['score']:.2%})")
```

### Manual Model Loading (Advanced)

If you need more control:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("brightertiger/garbled-text-detector")
model = AutoModelForSequenceClassification.from_pretrained("brightertiger/garbled-text-detector")

text = "Your text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()

label = model.config.id2label[prediction]
print(f"Prediction: {label} (confidence: {confidence:.2%})")
```

## What It Does

This model classifies text into two categories:

**Normal (Label 0)**: Clean, well-formed, readable text
- Example: *"The company reported revenue of $1.2 million in Q4."*

**Garbled (Label 1)**: Corrupted, malformed, or unreadable text
- Example: *"The%âãÏÓcompany%âãÏÓreported$1.2%âãÏÓmillion"*

### Common Use Cases

- **Data Quality**: Filter corrupted text from datasets before training
- **PDF Parsing**: Validate text extraction quality from PDFs
- **OCR Validation**: Check if OCR output is readable
- **Encoding Detection**: Identify character encoding issues
- **Content Filtering**: Remove malformed user-generated content

### What It Handles

✅ **Normal Text**: Including technical content with:
- Mathematical symbols: ∑, ∫, ±, ≤, α, β
- Currency symbols: $, €, £, ¥
- Legal citations and references
- Formatted tables and structured data

✅ **Detects Corruption**: 
- Binary data interpretation
- Encoding corruption
- Mixed character sets
- Non-ASCII noise

## Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.2% |
| **Precision** | 92.5% |
| **Recall** | 95.4% |
| **F1-Score** | 93.9% |
| **ROC-AUC** | 99.1% |

**Inference Performance** (CPU):
- Throughput: ~70 samples/second
- Latency: ~14ms per sample
- Model Size: 55MB
- Device: CPU optimized with dynamic quantization

**Confusion Matrix** (6,646 validation samples):
- True Negatives: 4,417 (Normal → Normal)
- False Positives: 171 (Normal → Garbled)
- False Negatives: 86 (Garbled → Normal)
- True Positives: 1,972 (Garbled → Garbled)

## Model Details

**Architecture**:
- Base Model: TinyBERT (huawei-noah/TinyBERT_General_4L_312D)
- Model Type: `AutoModelForSequenceClassification`
- Parameters: 14.4M
- Max Sequence Length: 512 tokens
- Task: Binary classification (Normal vs Garbled)

**Model Specifications**:
- Framework: PyTorch + Transformers
- Model Size: 55MB
- Input: Text (up to 512 tokens)
- Output: Binary classification with confidence scores
- Labels: `NORMAL` (0) and `GARBLED` (1)

**Training Data**:
- Synthetic data generated with LLMs simulating real-world PDF parsing scenarios
- Includes diverse corruption patterns and edge cases
- Balanced dataset with both normal and garbled examples

## Example Outputs

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="brightertiger/garbled-text-detector")

# Normal text
classifier("The quarterly earnings report shows strong growth.")
# → [{'label': 'NORMAL', 'score': 0.998}]

# Technical content (correctly identified as normal)
classifier("Calculate ROI = (Gain - Cost) / Cost × 100%")
# → [{'label': 'NORMAL', 'score': 0.973}]

# Currency symbols (correctly identified as normal)
classifier("Exchange rates: USD$1.00, EUR€0.85, GBP£0.73")
# → [{'label': 'NORMAL', 'score': 0.961}]

# Garbled text
classifier("PDF-1.4%âãÏÓ1 0 obj<< /Type /Catalog")
# → [{'label': 'GARBLED', 'score': 0.992}]

# Encoding corruption
classifier("Revenue%âãÏÓ$1,250,000%âãÏÓProfit%âãÏÓ$890,000")
# → [{'label': 'GARBLED', 'score': 0.987}]
```

## Integration Examples

### Data Cleaning Pipeline

```python
from transformers import pipeline

def clean_dataset(texts):
    classifier = pipeline("text-classification", model="brightertiger/garbled-text-detector")
    results = classifier(texts)
    
    clean_texts = [
        text for text, result in zip(texts, results)
        if result['label'] == "NORMAL" and result['score'] > 0.9
    ]
    
    return clean_texts
```

### Quality Control

```python
from transformers import pipeline

def validate_text_quality(text, threshold=0.85):
    classifier = pipeline("text-classification", model="brightertiger/garbled-text-detector")
    result = classifier(text)[0]
    
    if result['label'] == "GARBLED" and result['score'] > threshold:
        return {
            "status": "rejected",
            "reason": "garbled_text",
            "confidence": result['score']
        }
    
    return {"status": "approved", "confidence": result['score']}
```

### PDF Extraction Validation

```python
from transformers import pipeline

def validate_pdf_extraction(pdf_text):
    classifier = pipeline("text-classification", model="brightertiger/garbled-text-detector")
    chunks = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 1000)]
    
    results = classifier(chunks)
    garbled_count = sum(1 for r in results if r['label'] == "GARBLED")
    garbled_ratio = garbled_count / len(chunks)
    
    if garbled_ratio > 0.2:
        return {"quality": "poor", "garbled_ratio": garbled_ratio}
    
    return {"quality": "good", "garbled_ratio": garbled_ratio}
```

## API Reference

### Using with Transformers Pipeline

```python
from transformers import pipeline

# Load the model
classifier = pipeline(
    "text-classification", 
    model="brightertiger/garbled-text-detector"
)

# Single prediction
result = classifier("Your text here")[0]
print(f"{result['label']}: {result['score']:.2%}")

# Batch prediction
results = classifier(["Text 1", "Text 2", "Text 3"])
```

### Custom Device Selection

```python
from transformers import pipeline

# Use GPU if available
classifier = pipeline(
    "text-classification",
    model="brightertiger/garbled-text-detector",
    device=0  # Use GPU 0, or -1 for CPU
)

# Use CPU explicitly
classifier = pipeline(
    "text-classification",
    model="brightertiger/garbled-text-detector",
    device=-1
)
```

### Confidence Thresholding

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="brightertiger/garbled-text-detector")

def is_text_valid(text, confidence_threshold=0.85):
    result = classifier(text)[0]
    
    # Accept if classified as NORMAL with high confidence
    if result['label'] == 'NORMAL' and result['score'] > confidence_threshold:
        return True
    
    # Reject if classified as GARBLED with high confidence
    if result['label'] == 'GARBLED' and result['score'] > confidence_threshold:
        return False
    
    # Uncertain - manual review recommended
    return None  # Flag for manual review
```

## Limitations

- Optimized for English text
- May struggle with intentionally stylized text (e.g., l33t speak used artistically)
- Designed for PDF parsing validation; may not generalize to all corruption types
- Max sequence length: 512 tokens

## License

Apache 2.0

## Citation

```bibtex
@misc{garbled_text_detector_2025,
  title={Garbled Text Detector: BERT-based Binary Classifier for Text Quality Assessment},
  author={Ujjwal Singh Rao},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/brightertiger/garbled-text-detector}
}
```

## Acknowledgments

- [Hugging Face](https://huggingface.co) for model hosting and Transformers library
- [Huawei Noah's Ark Lab](https://huggingface.co/huawei-noah) for TinyBERT base model
- PyTorch Lightning for training framework

---

**Model Card**: brightertiger/garbled-text-detector  
**Version**: 1.0.0  
**Last Updated**: September 2025  
**Task**: Binary Text Classification

## Support & Issues

For questions, bug reports, or feature requests:
- 🐛 Open an issue on [GitHub](https://github.com/yourusername/pygarble-hf-models/issues)
- 💬 Discuss on the [Hugging Face Community](https://huggingface.co/brightertiger/garbled-text-detector/discussions)

## Contributing

Contributions are welcome! If you'd like to improve the model or add features, please open a pull request on GitHub.