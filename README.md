# NeuroFed — Multi-Hospital Federated Learning (Brain Tumor MRI)

Each teammate uses a different real brain tumor MRI dataset,
simulating a different hospital. One teammate acts as the server
and aggregates everyone's trained weights into a global model.

## Files

- `models/base_model.py` — shared ResNet-18 architecture (SAME for everyone)
- `models/fl_engine.py` — training and evaluation functions (SAME for everyone)
- `client_train.py` — run this on YOUR dataset (edit DATASET_PATH and CLIENT_NAME)

## Setup (Colab)

1. Download your dataset via Kaggle API into /content/data
2. `!git clone https://github.com/<your-username>/NeuroFed.git`
3. Edit `client_train.py`: set DATASET_PATH and CLIENT_NAME
4. Run `client_train.py`
5. Share the resulting `.pt` file with your server teammate

## Class consistency (IMPORTANT)

All teammates must map their folders to these 4 classes in this exact order:
glioma, meningioma, notumor, pituitary
