"""client.py - local training + honest evaluation for one client.

Fixes vs. the old client_train.py / notebook:
  * split is SEEDED, stratified, and saved (as file paths) inside the checkpoint
  * weights are SAVED (arch + classes + split included)
  * evaluation reuses the saved test split and asserts zero overlap with train/val
  * confusion-matrix title comes from the checkpoint's arch and the matrix itself

Usage:
  python client.py train --data ./data --arch resnet18 --epochs 10 --out client1.pt
  python client.py eval  --data ./data --ckpt client1.pt --plot client1_cm.png
"""
import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from base_model import get_base_model, save_checkpoint, load_checkpoint
from f1engine import F1Engine

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_split(root, seed=42, fractions=(0.7, 0.15, 0.15)):
    """Stratified, seeded split. Returns {"train":[relpaths], "val":[...], "test":[...]}."""
    ds = datasets.ImageFolder(root)
    rng = np.random.RandomState(seed)
    split = {"train": [], "val": [], "test": []}
    for cls_idx in range(len(ds.classes)):
        paths = sorted(os.path.relpath(p, root) for p, y in ds.samples if y == cls_idx)
        rng.shuffle(paths)
        n_train = int(len(paths) * fractions[0])
        n_val = int(len(paths) * fractions[1])
        split["train"] += paths[:n_train]
        split["val"] += paths[n_train:n_train + n_val]
        split["test"] += paths[n_train + n_val:]
    assert_no_leakage(split)
    return split


def assert_no_leakage(split):
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    assert not (tr & te), f"LEAK: {len(tr & te)} images in both train and test"
    assert not (va & te), f"LEAK: {len(va & te)} images in both val and test"
    assert not (tr & va), f"LEAK: {len(tr & va)} images in both train and val"


def make_loader(root, rel_paths, transform, batch_size, shuffle):
    ds = datasets.ImageFolder(root, transform=transform)
    lookup = {os.path.relpath(p, root): i for i, (p, _) in enumerate(ds.samples)}
    missing = [p for p in rel_paths if p not in lookup]
    if missing:
        raise FileNotFoundError(f"{len(missing)} split files not found in {root}, e.g. {missing[:3]}")
    return DataLoader(Subset(ds, [lookup[p] for p in rel_paths]),
                      batch_size=batch_size, shuffle=shuffle, num_workers=2), ds.classes


class Client:
    def __init__(self, data_root, arch="resnet18", seed=42, batch_size=32, lr=1e-3,
                 device=None, split=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.arch, self.seed, self.data_root = arch, seed, data_root
        set_seed(seed)
        self.split = split or make_split(data_root, seed)
        self.train_loader, self.classes = make_loader(data_root, self.split["train"], TRAIN_TF, batch_size, True)
        self.val_loader, _ = make_loader(data_root, self.split["val"], EVAL_TF, batch_size, False)
        self.test_loader, _ = make_loader(data_root, self.split["test"], EVAL_TF, batch_size, False)
        self.model = get_base_model(arch, len(self.classes)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # ---- federated hooks -------------------------------------------------
    def get_weights(self):
        return copy.deepcopy(self.model.state_dict())

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True)

    # ---- training --------------------------------------------------------
    def train(self, epochs=10):
        """Trains on TRAIN, picks the best epoch on VAL. TEST is never touched here."""
        best_acc, best_state = -1.0, None
        engine = F1Engine(self.classes)
        for ep in range(1, epochs + 1):
            self.model.train()
            running = 0.0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()
                running += loss.item() * x.size(0)
            val = engine.evaluate(self.model, self.val_loader, self.device)
            print(f"epoch {ep}/{epochs} loss={running / len(self.split['train']):.4f} "
                  f"val_acc={val['accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f}")
            if val["accuracy"] > best_acc:
                best_acc, best_state = val["accuracy"], self.get_weights()
        self.set_weights(best_state)
        return best_acc

    def evaluate_test(self):
        engine = F1Engine(self.classes)
        metrics = engine.evaluate(self.model, self.test_loader, self.device)
        return engine, metrics

    def save(self, path):
        save_checkpoint(path, self.model, self.arch, self.classes, split=self.split, seed=self.seed)
        print(f"saved -> {path}")


def evaluate_checkpoint(data_root, ckpt_path, plot_path=None, device=None):
    """Evaluate a saved checkpoint on the SAME test images that were held out in training."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(ckpt_path, device)
    if ckpt.get("split") is None:
        raise ValueError("Checkpoint has no saved split -> cannot prove a clean holdout. Retrain with client.py.")
    assert_no_leakage(ckpt["split"])
    loader, classes = make_loader(data_root, ckpt["split"]["test"], EVAL_TF, 32, False)
    assert classes == ckpt["classes"], "Class order differs from training"
    engine = F1Engine(classes)
    metrics = engine.evaluate(model, loader, device)
    if plot_path:
        engine.plot_confusion_matrix(plot_path, arch=ckpt["arch"])
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="client1.pt")
    ap.add_argument("--ckpt", default="client1.pt")
    ap.add_argument("--plot", default="client1_confusion_matrix.png")
    a = ap.parse_args()

    if a.mode == "train":
        c = Client(a.data, arch=a.arch, seed=a.seed)
        c.train(a.epochs)
        c.save(a.out)
        engine, m = c.evaluate_test()
        engine.plot_confusion_matrix(a.plot, arch=a.arch)
        print(json.dumps({k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")}, indent=2))
    else:
        m = evaluate_checkpoint(a.data, a.ckpt, a.plot)
        print(json.dumps({k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")}, indent=2))
        print("per class:", json.dumps(m["per_class"], indent=2))