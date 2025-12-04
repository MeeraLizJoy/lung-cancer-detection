"""
Model Evaluation Script
=======================
Evaluates the trained EfficientNet-B0 model on a labeled validation/test dataset.

This script:
- Loads the final trained model
- Runs predictions on a labeled dataset (e.g., data/valid/)
- Computes accuracy, precision, recall, F1-score
- Generates and saves confusion matrix visualization

Usage:
    python src/evaluate.py --data_dir data/valid/
"""

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

# Device configuration
device = torch.device("cpu")

# Class labels
CLASS_NAMES = ['bengin', 'malignant', 'normal']
NUM_CLASSES = 3
DROPOUT = 0.2

# Model path
MODEL_PATH = 'models/efficientnet_b0_final.pth'

def load_model(model_path=MODEL_PATH):
    """Load the trained EfficientNet-B0 model."""
    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(in_features, NUM_CLASSES)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from: {model_path}\n")
    return model

def evaluate_model(model, data_loader):
    """
    Run evaluation on a labeled dataset.
    
    Returns:
        y_true: List of true labels
        y_pred: List of predicted labels
        y_probs: List of prediction probabilities
    """
    y_true = []
    y_pred = []
    y_probs = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probabilities.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(y_probs)

def print_metrics(y_true, y_pred):
    """Print evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60)
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

def plot_confusion_matrix(y_true, y_pred, save_path='evaluation_results/confusion_matrix.png'):
    """Generate and save confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - EfficientNet-B0', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved: {save_path}")
    plt.show()

def plot_class_accuracies(y_true, y_pred, save_path='evaluation_results/class_accuracies.png'):
    """Plot per-class accuracy."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    class_accuracies = []
    for i in range(NUM_CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).sum() / mask.sum()
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(CLASS_NAMES, class_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.05])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Class accuracies plot saved: {save_path}")
    plt.show()

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained model on labeled dataset')
    parser.add_argument('--data_dir', type=str, default='data/valid/',
                       help='Path to labeled dataset directory (ImageFolder structure)')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help='Path to trained model weights')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please provide a valid path to a labeled dataset with ImageFolder structure.")
        return
    
    # Data transforms (same as training/validation)
    mean, std = [0.5], [0.5]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Classes: {dataset.classes}\n")
    
    # Load model
    model = load_model(args.model)
    
    # Run evaluation
    y_true, y_pred, y_probs = evaluate_model(model, data_loader)
    
    # Print metrics
    print_metrics(y_true, y_pred)
    
    # Generate visualizations
    plot_confusion_matrix(y_true, y_pred)
    plot_class_accuracies(y_true, y_pred)
    
    print("\n" + "="*60)
    print("Evaluation complete! Results saved in evaluation_results/")
    print("="*60)

if __name__ == "__main__":
    main()
