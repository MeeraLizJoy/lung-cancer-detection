"""
Lung Cancer Detection - Inference Script
=========================================
Loads the trained EfficientNet-B0 model and makes predictions on new CT scan images.

Usage:
    # Single image prediction
    python src/predict.py --image path/to/image.jpg
    
    # Batch prediction on folder (flat or nested)
    python src/predict.py --folder path/to/images/
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import os
import sys

# Device configuration
device = torch.device("cpu")

# Class labels
CLASS_NAMES = ['bengin', 'malignant', 'normal']
NUM_CLASSES = 3

# Best hyperparameters used during training
DROPOUT = 0.2

# Model path
MODEL_PATH = 'models/efficientnet_b0_final.pth'

def load_model(model_path=MODEL_PATH):
    """Load the trained EfficientNet-B0 model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize model architecture
    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(in_features, NUM_CLASSES)
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from: {model_path}\n")
    return model

def preprocess_image(image_path):
    """Preprocess image for model input."""
    mean, std = [0.5], [0.5]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_image(model, image_path, return_probs=True):
    """Predict class and confidence for a single image."""
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()
    
    result = {
        'image': os.path.basename(image_path),
        'predicted_class': predicted_class,
        'confidence': confidence_score
    }
    
    if return_probs:
        result['probabilities'] = {
            class_name: prob.item() 
            for class_name, prob in zip(CLASS_NAMES, probabilities)
        }
    
    return result

def find_all_images(folder_path):
    """Recursively find all image files in folder and subfolders."""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_extensions:
                image_files.append(os.path.join(root, f))
    
    return image_files

def predict_folder(model, folder_path):
    """Predict classes for all images in a folder (including subfolders)."""
    image_files = find_all_images(folder_path)
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return []
    
    print(f"Found {len(image_files)} images. Processing...\n")
    
    results = []
    for image_path in image_files:
        try:
            result = predict_image(model, image_path)
            results.append(result)
            print(f"✓ {result['image']}: {result['predicted_class']} "
                  f"({result['confidence']:.2%} confidence)")
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(image_path)}: {e}")
    
    return results

def print_prediction_details(result):
    """Pretty print prediction results."""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {result['image']}")
    print(f"Predicted Class: {result['predicted_class'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nClass Probabilities:")
    for class_name, prob in result['probabilities'].items():
        bar = '█' * int(prob * 40)
        print(f"  {class_name:12s}: {prob:.2%} {bar}")
    print("="*60)

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Lung Cancer Detection - Model Inference'
    )
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--model', type=str, default=MODEL_PATH, 
                       help='Path to model weights')
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.print_help()
        sys.exit(1)
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        
        result = predict_image(model, args.image)
        print_prediction_details(result)
    
    # Batch prediction (handles nested folders)
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            sys.exit(1)
        
        results = predict_folder(model, args.folder)
        
        if results:
            print(f"\n{'='*60}")
            print(f"SUMMARY: Processed {len(results)} images")
            print(f"{'='*60}")

if __name__ == "__main__":
    main()
