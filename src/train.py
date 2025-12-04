import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import time
import csv
import matplotlib.pyplot as plt
from data_loader import train_loader, valid_loader, class_names  # Your data loaders

device = torch.device("cpu")

model_dict = {
    "resnet18": models.resnet18,
    "vgg16": models.vgg16,
    "efficientnet_b0": models.efficientnet_b0,
    "densenet121": models.densenet121,
}

def initialize_model(model_name, num_classes):
    model = model_dict[model_name](pretrained=True)
    if model_name.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith('vgg'):
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name.startswith('efficientnet'):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name.startswith('densenet'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Unknown model_name")
    return model.to(device)

def plot_metrics(log_path, model_name):
    epochs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            train_accs.append(float(row['train_acc']))
            val_accs.append(float(row['val_acc']))

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.legend()
    plt.title(f"{model_name} Loss")

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.legend()
    plt.title(f"{model_name} Accuracy")
    plt.show()

def train(model_name, num_classes, epochs=10, save_dir='checkpoints', log_dir='logs'):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = initialize_model(model_name, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0

    log_path = os.path.join(log_dir, f"{model_name}_training_log.csv")
    with open(log_path, mode='w', newline='') as log_file:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'epoch_duration']
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(epochs):
            start_time = time.time()
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
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_loss_avg = val_loss / val_total
            val_acc = val_correct / val_total
            epoch_duration = time.time() - start_time

            print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                  f"Val loss: {val_loss_avg:.4f}, acc: {val_acc:.4f} - Duration: {epoch_duration:.1f}s")

            writer.writerow({'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc,
                             'val_loss': val_loss_avg, 'val_acc': val_acc, 'epoch_duration': epoch_duration})

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pth"))

    # Plot metrics after training completes
    plot_metrics(log_path, model_name)

if __name__ == "__main__":
    print("Classes:", class_names)
    num_classes = len(class_names)
    models_to_train = ["resnet18", "vgg16", "efficientnet_b0", "densenet121"]
    for model_name in models_to_train:
        print(f"Training model: {model_name}")
        train(model_name, num_classes, epochs=10)
