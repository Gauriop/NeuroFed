"""
CLIENT TRAINING SCRIPT — full pipeline
EDA -> Cleaning -> Architecture Comparison (with early stopping) -> Save best model
"""

import torch
import os
import sys
import json
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('/content/NeuroFed')
from models.base_model import get_base_model
from models.fl_engine import train_local_model_with_early_stopping, evaluate_model

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

# ============================================================
# CONFIG — change these for your setup
# ============================================================
DATASET_PATH = "/content/data/Training"
CLIENT_NAME = "gauri_kaggle"
ARCHITECTURES_TO_COMPARE = ["resnet18", "vgg16", "mobilenet_v2"]
IMG_SIZE = 128
RESULTS_DIR = "/content/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

classes = sorted(os.listdir(DATASET_PATH))
print("Classes found:", classes)

# ============================================================
# STEP 1: EDA — before any cleaning, see what we actually have
# ============================================================
print("\n" + "="*60)
print("STEP 1: EXPLORATORY DATA ANALYSIS")
print("="*60)

# 1a. Class counts table
counts = [len(os.listdir(os.path.join(DATASET_PATH, c))) for c in classes]
eda_table = pd.DataFrame({"Class": classes, "Count": counts})
print(eda_table)
eda_table.to_csv(f"{RESULTS_DIR}/{CLIENT_NAME}_eda_class_counts.csv", index=False)

plt.figure(figsize=(7,5))
plt.bar(classes, counts, color="#1FB2A6")
plt.title(f"Class Distribution — {CLIENT_NAME}")
plt.ylabel("Image Count")
for i, v in enumerate(counts):
    plt.text(i, v+10, str(v), ha="center")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/{CLIENT_NAME}_fig1_class_distribution.png", dpi=150)
plt.close()

# 1b. Sample images grid
fig, axes = plt.subplots(1, len(classes), figsize=(4*len(classes), 4))
for ax, cls in zip(axes, classes):
    cls_path = os.path.join(DATASET_PATH, cls)
    img = Image.open(os.path.join(cls_path, random.choice(os.listdir(cls_path))))
    ax.imshow(img, cmap="gray")
    ax.set_title(cls)
    ax.axis("off")
plt.suptitle(f"Sample Images — {CLIENT_NAME}")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/{CLIENT_NAME}_fig2_sample_images.png", dpi=150)
plt.close()

# 1c. Image dimension check
widths, heights = [], []
for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    for fname in random.sample(os.listdir(cls_path), min(50, len(os.listdir(cls_path)))):
        try:
            img = Image.open(os.path.join(cls_path, fname))
            widths.append(img.size[0])
            heights.append(img.size[1])
        except Exception:
            pass
print(f"Width range: {min(widths)}-{max(widths)}, Height range: {min(heights)}-{max(heights)}")

print("EDA figures saved:", f"{CLIENT_NAME}_fig1_class_distribution.png, {CLIENT_NAME}_fig2_sample_images.png")

# ============================================================
# STEP 2: CLEANING — remove corrupt/unreadable images
# ============================================================
print("\n" + "="*60)
print("STEP 2: DATA CLEANING")
print("="*60)

corrupt_files = []
for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    for fname in os.listdir(cls_path):
        fpath = os.path.join(cls_path, fname)
        try:
            img = Image.open(fpath)
            img.verify()
        except Exception:
            corrupt_files.append(fpath)

print(f"Corrupt/unreadable files found: {len(corrupt_files)}")
for f in corrupt_files:
    print("  Removing:", f)
    os.remove(f)

print("Cleaning complete. Dataset is now safe to load.")

# ============================================================
# STEP 3: Build dataset objects (post-cleaning)
# ============================================================
print("\n" + "="*60)
print("STEP 3: BUILDING DATASET")
print("="*60)

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = ImageFolder(DATASET_PATH, transform=transform)
print("Class mapping:", full_dataset.class_to_idx)
print("Total usable images after cleaning:", len(full_dataset))

n = len(full_dataset)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
test_size = n - train_size - val_size
train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])
print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# ============================================================
# STEP 4: Train + compare architectures (with early stopping)
# ============================================================
print("\n" + "="*60)
print("STEP 4: ARCHITECTURE COMPARISON")
print("="*60)

results = {}

for arch in ARCHITECTURES_TO_COMPARE:
    print(f"\n--- Training: {arch} ---")
    model = get_base_model(arch=arch, num_classes=len(classes), pretrained=True)

    model, history = train_local_model_with_early_stopping(
        model, train_data, val_data, device,
        max_epochs=20, patience=3
    )

    metrics = evaluate_model(model, test_data, device)
    print(f"{arch} TEST metrics: {metrics}")