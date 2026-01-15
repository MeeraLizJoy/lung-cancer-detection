import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Paths to train and validation directories
train_dir = "data/train"
valid_dir = "data/valid"

# Normalization parameters (mean and std for grayscale or RGB)
mean, std = [0.5], [0.5]

# Training transforms (with augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # Ensure all images are 224x224
    transforms.RandomHorizontalFlip(),      # Data augmentation: horizontal flip
    transforms.RandomRotation(10),          # Data augmentation: rotation up to 10 degrees
    transforms.ToTensor(),                  # Convert PIL image to tensor
    transforms.Normalize(mean, std),        # Normalize pixel values
])

# Validation transforms (no augmentation, only preprocessing)
valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # Ensure all images are 224x224
    transforms.ToTensor(),                  # Convert PIL image to tensor
    transforms.Normalize(mean, std),        # Normalize pixel values
])

# Create PyTorch ImageFolder datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Create DataLoaders
# num_workers=0 for CPU/Mac stability; increase to 2-4 if using GPU and no issues
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)

# Extract class names for reference
class_names = train_dataset.classes
print("Classes:", class_names)
