import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import time
import csv
from itertools import product
from data_loader import train_loader, valid_loader, class_names

device = torch.device("cpu")

model_dict = {
    "resnet18": models.resnet18,
    "vgg16": models.vgg16,
    "efficientnet_b0": models.efficientnet_b0,
    "densenet121": models.densenet121,
}

def initialize_model(model_name, num_classes, dropout_rate=0.0):
    """Initialize model with optional dropout for regularization."""
    model = model_dict[model_name](pretrained=True)
    
    if model_name.startswith('resnet'):
        # Add dropout before final layer
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

def train_with_params(model_name, num_classes, lr, weight_decay, dropout, 
                      epochs=10, save_dir='tuning_checkpoints', log_dir='tuning_logs'):
    """Train model with specific hyperparameters."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = initialize_model(model_name, num_classes, dropout_rate=dropout)
    criterion = nn.CrossEntropyLoss()
    
    # Use Adam optimizer with weight decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0.0
    config_name = f"{model_name}_lr{lr}_wd{weight_decay}_drop{dropout}"
    
    log_path = os.path.join(log_dir, f"{config_name}.csv")
    
    with open(log_path, mode='w', newline='') as log_file:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'time']
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(epochs):
            start = time.time()
            
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

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss_avg = val_loss / val_total
            val_acc = val_correct / val_total
            epoch_time = time.time() - start

            writer.writerow({
                'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss_avg, 'val_acc': val_acc, 'time': epoch_time
            })

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 
                          os.path.join(save_dir, f"{config_name}_best.pth"))

    return best_val_acc

def grid_search(model_name, num_classes, hyperparams, epochs=10):
    """Perform grid search over hyperparameters."""
    results = []
    
    # Generate all combinations
    keys = hyperparams.keys()
    combinations = list(product(*hyperparams.values()))
    
    print(f"\n{'='*60}")
    print(f"Starting Grid Search for {model_name}")
    print(f"Total configurations to test: {len(combinations)}")
    print(f"{'='*60}\n")
    
    for i, values in enumerate(combinations, 1):
        params = dict(zip(keys, values))
        print(f"[{i}/{len(combinations)}] Testing config: {params}")
        
        best_val_acc = train_with_params(
            model_name, num_classes,
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            dropout=params['dropout'],
            epochs=epochs
        )
        
        results.append({
            'model': model_name,
            'lr': params['lr'],
            'weight_decay': params['weight_decay'],
            'dropout': params['dropout'],
            'best_val_acc': best_val_acc
        })
        
        print(f"Best Val Acc: {best_val_acc:.4f}\n")
    
    return results

def save_tuning_results(all_results, filename='tuning_results.csv'):
    """Save all tuning results to CSV."""
    with open(filename, 'w', newline='') as f:
        fieldnames = ['model', 'lr', 'weight_decay', 'dropout', 'best_val_acc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    print(f"\nAll results saved to {filename}")

if __name__ == "__main__":
    print("Classes:", class_names)
    num_classes = len(class_names)
    
    # Define hyperparameter search spaces based on baseline performance
    
    # EfficientNet-B0: Already good, fine-tune around baseline
    efficientnet_params = {
        'lr': [0.001, 0.0005, 0.0001],
        'weight_decay': [0.0, 0.0001, 0.001],
        'dropout': [0.0, 0.2]
    }
    
    # ResNet18 & DenseNet121: Need regularization to reduce overfitting
    regularization_params = {
        'lr': [0.0001, 0.00005],
        'weight_decay': [0.001, 0.01],
        'dropout': [0.3, 0.5]
    }
    
    # VGG16: Try very low learning rates to get it to learn
    vgg_params = {
        'lr': [0.00001, 0.000001],
        'weight_decay': [0.0, 0.0001],
        'dropout': [0.0, 0.2]
    }
    
    all_results = []
    
    # Tune EfficientNet-B0 (best performer)
    results = grid_search("efficientnet_b0", num_classes, efficientnet_params, epochs=8)
    all_results.extend(results)
    
    # Tune ResNet18
    results = grid_search("resnet18", num_classes, regularization_params, epochs=8)
    all_results.extend(results)
    
    # Tune DenseNet121
    results = grid_search("densenet121", num_classes, regularization_params, epochs=8)
    all_results.extend(results)
    
    # Tune VGG16
    results = grid_search("vgg16", num_classes, vgg_params, epochs=8)
    all_results.extend(results)
    
    # Save all results
    save_tuning_results(all_results)
    
    # Print best configurations
    print("\n" + "="*60)
    print("BEST CONFIGURATIONS:")
    print("="*60)
    for model in ["efficientnet_b0", "resnet18", "densenet121", "vgg16"]:
        model_results = [r for r in all_results if r['model'] == model]
        best = max(model_results, key=lambda x: x['best_val_acc'])
        print(f"\n{model}:")
        print(f"  LR: {best['lr']}, Weight Decay: {best['weight_decay']}, Dropout: {best['dropout']}")
        print(f"  Best Val Acc: {best['best_val_acc']:.4f}")
