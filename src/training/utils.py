import os
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly converted
    if 'training' in config:
        training = config['training']
        if 'learning_rate' in training:
            training['learning_rate'] = float(training['learning_rate'])
        if 'weight_decay' in training:
            training['weight_decay'] = float(training['weight_decay'])
        if 'warmup_steps' in training:
            training['warmup_steps'] = int(training['warmup_steps'])
        if 'num_epochs' in training:
            training['num_epochs'] = int(training['num_epochs'])
        if 'batch_size' in training:
            training['batch_size'] = int(training['batch_size'])
        if 'gradient_clip_val' in training:
            training['gradient_clip_val'] = float(training['gradient_clip_val'])
        if 'accumulate_grad_batches' in training:
            training['accumulate_grad_batches'] = int(training['accumulate_grad_batches'])
    
    if 'model' in config:
        model = config['model']
        if 'max_length' in model:
            model['max_length'] = int(model['max_length'])
        if 'dropout' in model:
            model['dropout'] = float(model['dropout'])
    
    if 'hardware' in config:
        hardware = config['hardware']
        if 'num_workers' in hardware:
            hardware['num_workers'] = int(hardware['num_workers'])
        if 'devices' in hardware and hardware['devices'] != 'auto' and hardware['devices'] != 'cpu':
            hardware['devices'] = int(hardware['devices'])
    
    if 'logging' in config:
        logging_config = config['logging']
        if 'log_every_n_steps' in logging_config:
            logging_config['log_every_n_steps'] = int(logging_config['log_every_n_steps'])
    
    if 'data' in config:
        data = config['data']
        if 'val_split' in data:
            data['val_split'] = float(data['val_split'])
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_metrics(trainer, save_path: Optional[str] = None):
    """Plot training metrics from trainer logs."""
    if not hasattr(trainer, 'logged_metrics'):
        print("No logged metrics found.")
        return
    
    metrics = trainer.logged_metrics
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot loss
    if 'train_loss' in metrics and 'val_loss' in metrics:
        axes[0, 0].plot(metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(metrics['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
    
    # Plot accuracy
    if 'train_acc' in metrics and 'val_acc' in metrics:
        axes[0, 1].plot(metrics['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(metrics['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
    
    # Plot F1 score
    if 'train_f1' in metrics and 'val_f1' in metrics:
        axes[1, 0].plot(metrics['train_f1'], label='Train F1')
        axes[1, 0].plot(metrics['val_f1'], label='Validation F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
    
    # Plot precision and recall
    if 'train_precision' in metrics and 'val_precision' in metrics:
        axes[1, 1].plot(metrics['train_precision'], label='Train Precision')
        axes[1, 1].plot(metrics['val_precision'], label='Validation Precision')
        axes[1, 1].plot(metrics['train_recall'], label='Train Recall')
        axes[1, 1].plot(metrics['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Training and Validation Precision/Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=['Normal', 'Garbled'], save_path: Optional[str] = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def print_classification_report(y_true, y_pred, class_names=['Normal', 'Garbled']):
    """Print detailed classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)


def save_predictions(predictions, labels, texts, output_path: str):
    """Save predictions to CSV file."""
    import pandas as pd
    
    df = pd.DataFrame({
        'text': texts,
        'true_label': labels,
        'predicted_label': predictions
    })
    
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def load_model_checkpoint(checkpoint_path: str, model_class, **kwargs):
    """Load model from checkpoint."""
    model = model_class.load_from_checkpoint(checkpoint_path, **kwargs)
    return model


def create_directories(base_path: str, subdirs: list):
    """Create directory structure."""
    for subdir in subdirs:
        dir_path = os.path.join(base_path, subdir)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
