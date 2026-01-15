import os
import shutil
import random


RAW_DIR = "data/preprocessed"
TARGET_DIRS = ["data/train", "data/valid"]
SPLIT_RATIOS = [0.85, 0.15]
CLASSES = ['bengin', 'malignant', 'normal'] 


# To ensure target directories exist
for split in TARGET_DIRS:
    for label in CLASSES:
        os.makedirs(os.path.join(split, label), exist_ok=True)


# function to split and copy from raw to target directories
def split_and_copy(class_name):
    img_dir = os.path.join(RAW_DIR, class_name)
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    n_total = len(images)
    n_train = int(SPLIT_RATIOS[0] * n_total)

    splits = {
        'data/train': images[:n_train],
        'data/valid': images[n_train:]
    }

    for split_dir, split_imgs in splits.items():
        for img in split_imgs:
            src = os.path.join(img_dir, img)
            dst = os.path.join(split_dir, class_name, img)
            shutil.copyfile(src, dst)

if __name__ == '__main__':
    random.seed(42)
    for cls in CLASSES:
        split_and_copy(cls)
    print("Train/Valid split complete! data/train and data/valid folders are ready.")