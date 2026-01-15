import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cpu")
NUM_CLASSES = 3
VALID_DIR = 'data/valid'  # We evaluate on the Validation set
MODEL_PATH = 'models/efficientnet_b0_final.pth'
LOG_PATH = 'final_training_logs/final_training_log.csv'

# --- 2. LOAD DATA & MODEL ---
def get_model():
    model = models.efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    return model.to(DEVICE).eval()

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.ImageFolder(VALID_DIR, transform=valid_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# --- 3. GENERATE CURVES ---
def plot_learning_curves():
    df = pd.read_csv(LOG_PATH)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

# --- 4. GENERATE CONFUSION MATRIX ---
def plot_confusion_matrix(model):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

if __name__ == "__main__":
    net = get_model()
    plot_learning_curves()
    plot_confusion_matrix(net)