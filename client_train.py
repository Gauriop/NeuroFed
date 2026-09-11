"""
CLIENT TRAINING SCRIPT
Run this on YOUR dataset. Each teammate runs this same script,
only changing DATASET_PATH and CLIENT_NAME to their own.
"""

import torch
import os
import sys
sys.path.append('/content/NeuroFed')

from models.base_model import get_base_model
from models.fl_engine import train_local_model, evaluate_model
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

# ---- CHANGE THESE FOR YOUR DATASET ----
DATASET_PATH = "/content/data/Training"
CLIENT_NAME = "client1"
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

IMG_SIZE = 128
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = ImageFolder(DATASET_PATH, transform=transform)
print("Classes found:", full_dataset.class_to_idx)
print("Total images:", len(full_dataset))

# Split into local train/test (80/20) for your own evaluation
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
local_train, local_test = random_split(full_dataset, [train_size, test_size])

print(f"Local train: {len(local_train)} | Local test: {len(local_test)}")

# Train
model = get_base_model(num_classes=4, pretrained=True)
print("\nTraining local model...")
model = train_local_model(model, local_train, device, epochs=5)

# Evaluate on your own held-out data
metrics = evaluate_model(model, local_test, device)
print("\nLocal model performance (on own data):", metrics)

# Save weights
os.makedirs("/content/results", exist_ok=True)
save_path = f"/content/results/{CLIENT_NAME}_weights.pt"
torch.save(model.state_dict(), save_path)
print(f"\nSaved weights to {save_path}")
print("Share this .pt file with your team's 'server' person for aggregation.")