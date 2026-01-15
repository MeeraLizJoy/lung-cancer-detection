"""
Final Production Model Training Script
=======================================
Trains EfficientNet-B0 on the FULL preprocessed dataset using the best hyperparameters
discovered during hyperparameter tuning and validated through k-fold cross-validation.

Best Hyperparameters:
- Learning Rate: 0.0005
- Weight Decay: 0.0001
- Dropout: 0.2
- Expected Accuracy: ~99.6%

This model will be saved for deployment/inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
import time
import csv

# Device configuration
device = torch.device("cpu")

# Best hyperparameters from tuning
BEST_LR = 0.0005
BEST_WEIGHT_DECAY = 0.0001
BEST_DROPOUT = 0.2
NUM_CLASSES = 3
EPOCHS = 15  # More epochs since we're training on all data
BATCH_SIZE = 32

# Paths
PREPROCESSED_DIR = 'data/preprocessed'
MODEL_SAVE_DIR = 'models'
LOG_SAVE_DIR = 'final_training_logs'

# Create directories
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_SAVE_DIR, exist_ok=True)

def initialize_efficientnet_b0(num_classes, dropout_rate):
    """
    Initialize EfficientNet-B0 with custom classification head and dropout.
    
    Args:
        num_classes: Number of output classes (3: benign, malignant, normal)
        dropout_rate: Dropout probability for regularization
    
    Returns:
        model: Initialized EfficientNet-B0 model
    """
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(pretrained=True)
    
    # Replace classifier with custom head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    
    return model.to(device)

def train_final_model():
    """Train the final production model on all available data."""
    
    print("\n" + "="*70)
    print("TRAINING FINAL PRODUCTION MODEL: EfficientNet-B0")
    print("="*70)
    print(f"Learning Rate: {BEST_LR}")
    print(f"Weight Decay: {BEST_WEIGHT_DECAY}")
    print(f"Dropout: {BEST_DROPOUT}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # Data transforms (same as during validation)
    mean, std = [0.5], [0.5]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Light augmentation
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load FULL preprocessed dataset (no train/val split)
    print("Loading full dataset from:", PREPROCESSED_DIR)
    full_dataset = datasets.ImageFolder(PREPROCESSED_DIR, transform=transform)
    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}\n")
    
    # Create DataLoader
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    
    # Initialize model
    print("Initializing EfficientNet-B0 model...")
    model = initialize_efficientnet_b0(NUM_CLASSES, BEST_DROPOUT)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY)
    
    # Training log
    log_path = os.path.join(LOG_SAVE_DIR, 'final_training_log.csv')
    with open(log_path, 'w', newline='') as log_file:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'time']
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()
        
        print("Starting training...\n")
        best_acc = 0.0
        
        for epoch in range(EPOCHS):
            start_time = time.time()
            
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")
            
            # Epoch statistics
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
            print(f"  Train Loss: {epoch_loss:.4f}")
            print(f"  Train Accuracy: {epoch_acc:.4f} ({100*epoch_acc:.2f}%)")
            print(f"  Time: {epoch_time:.1f}s")
            print("-" * 70 + "\n")
            
            # Log to CSV
            writer.writerow({
                'epoch': epoch+1,
                'train_loss': epoch_loss,
                'train_acc': epoch_acc,
                'time': epoch_time
            })
            
            # Save best model (based on training accuracy since we have no validation set)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                model_path = os.path.join(MODEL_SAVE_DIR, 'efficientnet_b0_final.pth')
                torch.save(model.state_dict(), model_path)
                print(f"✓ Best model saved: {model_path} (Accuracy: {epoch_acc:.4f})\n")
    
    print("="*70)
    print("TRAINING COMPLETE!")
    print(f"Best Training Accuracy: {best_acc:.4f} ({100*best_acc:.2f}%)")
    print(f"Final model saved at: {model_path}")
    print(f"Training log saved at: {log_path}")
    print("="*70)

if __name__ == "__main__":
    # Record total training time
    total_start = time.time()
    
    # Train the model
    train_final_model()
    
    # Print total time
    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print("\nYour production model is ready for deployment!")
