# NeuroFed — Federated Learning on Brain Tumor MRI

## Structure

- `data/client_split.py` — IID / Non-IID client splitting
- `models/base_model.py` — shared ResNet-18 architecture
- `models/fl_engine.py` — core FedAvg training loop
- `notebooks/` — Colab-run scripts
- `results/` — output metrics and plots

## Setup (Colab)

1. `!git clone <your-repo-url>`
2. `%cd NeuroFed`
3. `!pip install -r requirements.txt -q`
4. Run `notebooks/00_setup_and_data.py` cell-by-cell
