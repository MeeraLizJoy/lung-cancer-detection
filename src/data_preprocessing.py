import os
from PIL import Image, ImageOps
import numpy as np
import random
from tqdm import tqdm


RAW_DIR = 'data/raw'
PREPROCESSED_DIR = 'data/preprocessed'
TARGET_SIZE = (224, 224)
AUGMENT_COUNT_TARGET = 400  # since size of the largest class


# Basic augmentations for the minority class (benign)
def augment_image(img):
    aug_imgs = []
    # Horizontal flip
    aug_imgs.append(ImageOps.mirror(img))
    # Vertical flip
    aug_imgs.append(ImageOps.flip(img))
    # 90 degree rotation
    aug_imgs.append(img.rotate(90))
    # 270 degree rotation
    aug_imgs.append(img.rotate(270))
    # Add more augmentations as needed
    return aug_imgs

os.makedirs(PREPROCESSED_DIR, exist_ok=True)


# Analyze image counts per class
class_counts = {cls: len(os.listdir(os.path.join(RAW_DIR, cls))) for cls in os.listdir(RAW_DIR)}

for cls in os.listdir(RAW_DIR):
    src_dir = os.path.join(RAW_DIR, cls)
    dst_dir = os.path.join(PREPROCESSED_DIR, cls)
    os.makedirs(dst_dir, exist_ok=True)
    fnames = os.listdir(src_dir)
    current_count = 0
    print(f"\nProcessing class: {cls}")

    # Copy and resize originals
    for fname in tqdm(fnames, desc=f"Resizing {cls}"):
        img_path = os.path.join(src_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(TARGET_SIZE)
        img.save(os.path.join(dst_dir, f'{current_count:04d}.jpg'))
        current_count += 1

    # Augmentation if this is the minority class
    if cls == 'bengin':
        print(f"  Augmenting {cls} to reach at least {AUGMENT_COUNT_TARGET} images...")
        base_imgs = [Image.open(os.path.join(dst_dir, f"{i:04d}.jpg")) for i in range(current_count)]
        while current_count < AUGMENT_COUNT_TARGET:
            img = random.choice(base_imgs)
            for aug in augment_image(img):
                if current_count < AUGMENT_COUNT_TARGET:
                    aug.save(os.path.join(dst_dir, f'{current_count:04d}.jpg'))
                    current_count += 1
                else:
                    break
        print(f"  Final {cls} images: {current_count}")
    else:
        print(f"  Final {cls} images: {current_count}")
