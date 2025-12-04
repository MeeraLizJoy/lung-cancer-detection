import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import csv
import time

device = torch.device("cpu")

# Model dictionary
model_dict = {
    "resnet18": models.resnet18,
    "vgg16": models.vgg16,
    "efficientnet_b0": models.efficientnet_b0,
    "densenet121": models.densenet121,
}

def initialize_model(model_name, num_classes, dropout_rate=0.0):
    """Initialize model with optional dropout."""
    model = model_dict[model_name](pretrained=True)
    
    if model_name.startswith('resnet'):
        if dropout_rate > 0:
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(model.fc.in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith('vgg'):
        if dropout_rate > 0:
            model.classifier[6] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(model.classifier[6].in_features, num_classes)
            )
        else:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name.startswith('efficientnet'):
        if dropout_rate > 0:
            model.classifier[1] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
        else:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name.startswith('densenet'):
        if dropout_rate > 0:
            model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(model.classifier.in_features, num_classes)
            )
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    return model.to(device)

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    """Train model for one fold and return validation metrics."""
    best_val_acc = 0.0
    fold_train_losses, fold_val_losses = [], []
    fold_train_accs, fold_val_accs = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        fold_train_losses.append(train_loss)
        fold_train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss_avg = val_loss / val_total
        val_acc = val_correct / val_total
        fold_val_losses.append(val_loss_avg)
        fold_val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc, fold_train_accs, fold_val_accs, fold_train_losses, fold_val_losses

def kfold_cross_validation(model_name, dataset, num_classes, hyperparams, 
                           n_splits=5, epochs=10, save_dir='kfold_results'):
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        model_name: Name of the model architecture
        dataset: Full dataset (ImageFolder)
        num_classes: Number of output classes
        hyperparams: Dict with 'lr', 'weight_decay', 'dropout'
        n_splits: Number of folds (default 5)
        epochs: Training epochs per fold
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract labels for stratification
    targets = np.array([label for _, label in dataset])
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    all_val_accs = []
    
    print(f"\n{'='*60}")
    print(f"Starting {n_splits}-Fold Cross-Validation for {model_name}")
    print(f"Hyperparameters: {hyperparams}")
    print(f"{'='*60}\n")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
        print(f"Fold {fold_idx}/{n_splits}")
        
        # Create train and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
        
        # Initialize model fresh for each fold
        model = initialize_model(model_name, num_classes, dropout_rate=hyperparams['dropout'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), 
                              lr=hyperparams['lr'], 
                              weight_decay=hyperparams['weight_decay'])
        
        # Train this fold
        start_time = time.time()
        best_val_acc, train_accs, val_accs, train_losses, val_losses = train_one_fold(
            model, train_loader, val_loader, criterion, optimizer, epochs
        )
        fold_time = time.time() - start_time
        
        print(f"  Best Val Acc: {best_val_acc:.4f}, Time: {fold_time:.1f}s\n")
        
        all_val_accs.append(best_val_acc)
        fold_results.append({
            'fold': fold_idx,
            'best_val_acc': best_val_acc,
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1],
            'time': fold_time
        })
    
    # Calculate mean and std across folds
    mean_val_acc = np.mean(all_val_accs)
    std_val_acc = np.std(all_val_accs)
    
    print(f"\n{'='*60}")
    print(f"K-Fold Results for {model_name}:")
    print(f"Mean Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in all_val_accs]}")
    print(f"{'='*60}\n")
    
    # Save results to CSV
    log_path = os.path.join(save_dir, f"{model_name}_kfold_results.csv")
    with open(log_path, 'w', newline='') as f:
        fieldnames = ['fold', 'best_val_acc', 'final_train_acc', 'final_val_acc', 'time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in fold_results:
            writer.writerow(result)
    
    # Save summary
    summary_path = os.path.join(save_dir, f"{model_name}_kfold_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Hyperparameters: {hyperparams}\n")
        f.write(f"Number of Folds: {n_splits}\n")
        f.write(f"Epochs per Fold: {epochs}\n\n")
        f.write(f"Mean Validation Accuracy: {mean_val_acc:.4f}\n")
        f.write(f"Std Validation Accuracy: {std_val_acc:.4f}\n")
        f.write(f"Individual Fold Accuracies: {all_val_accs}\n")
    
    return mean_val_acc, std_val_acc, fold_results

if __name__ == "__main__":
    # Data transforms
    mean, std = [0.5], [0.5]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load full preprocessed dataset (combining train and valid)
    # We'll use stratified k-fold to create our own splits
    full_dataset = datasets.ImageFolder('data/preprocessed', transform=transform)
    num_classes = len(full_dataset.classes)
    print("Classes:", full_dataset.classes)
    print(f"Total samples: {len(full_dataset)}\n")
    
    # Best hyperparameters from your tuning results
    best_configs = {
        "efficientnet_b0": {'lr': 0.0005, 'weight_decay': 0.0001, 'dropout': 0.2},
        "resnet18": {'lr': 5e-05, 'weight_decay': 0.001, 'dropout': 0.5},
        "densenet121": {'lr': 0.0001, 'weight_decay': 0.001, 'dropout': 0.3},
        "vgg16": {'lr': 1e-05, 'weight_decay': 0.0, 'dropout': 0.0},
    }
    
    # Run k-fold validation for each model
    all_results = {}
    for model_name, hyperparams in best_configs.items():
        mean_acc, std_acc, fold_results = kfold_cross_validation(
            model_name, 
            full_dataset, 
            num_classes, 
            hyperparams,
            n_splits=5,
            epochs=10
        )
        all_results[model_name] = {
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'fold_results': fold_results
        }
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL K-FOLD VALIDATION COMPARISON:")
    print("="*60)
    for model_name, results in all_results.items():
        print(f"{model_name}: {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")



# What This Does:

# Loads your full preprocessed dataset (from data/preprocessed).

# Splits it into 5 stratified folds preserving class distributions.

# Trains each model with its best hyperparameters on each fold.

# Reports mean and standard deviation of validation accuracy across folds.

# Saves detailed results per fold and summary files in kfold_results/.

# Expected Output:

# You'll get mean validation accuracy ± std for each model, which is a much more reliable performance estimate than a single train-validation split.

# This will help confirm whether your 100% accuracy results are robust or specific to the train-validation split you used earlier.

