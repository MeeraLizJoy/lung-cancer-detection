import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse

# Setup
device = torch.device("cpu")
CLASS_NAMES = ['bengin', 'malignant', 'normal']

def get_gradcam(model, input_tensor, target_layer):
    """Generates Grad-CAM heatmap."""
    feature_blobs = []
    backward_gradients = []

    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())
    def hook_gradient(module, grad_input, grad_output):
        backward_gradients.append(grad_output[0].cpu().data.numpy())

    # In EfficientNet-B0, 'features.8' is the last conv layer
    handle_f = target_layer.register_forward_hook(hook_feature)
    handle_g = target_layer.register_full_backward_hook(hook_gradient)

    # Forward pass
    output = model(input_tensor)
    idx = torch.argmax(output).item()
    
    # Backward pass
    model.zero_grad()
    output[0, idx].backward()

    # Generate Heatmap
    grads = backward_gradients[0]
    features = feature_blobs[0]
    weights = np.mean(grads, axis=(2, 3))[0]
    
    cam = np.zeros(features.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * features[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    handle_f.remove()
    handle_g.remove()
    return cam, idx

def visualize(image_path, model_path='models/efficientnet_b0_final.pth'):
    # Load Model
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 3))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Prep Image
    img_pil = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    input_tensor = transform(img_pil).unsqueeze(0)

    # Run Grad-CAM
    target_layer = model.features[8] # Final conv layer for EfficientNet-B0
    heatmap, pred_idx = get_gradcam(model, input_tensor, target_layer)

    # Plot
    img_np = np.array(img_pil.resize((224, 224)))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original: {os.path.basename(image_path)}")
    plt.imshow(img_np)
    plt.subplot(1, 2, 2)
    plt.title(f"AI Focus: {CLASS_NAMES[pred_idx]}")
    plt.imshow(overlayed)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    visualize(args.image)