"""base_model.py - model factory + checkpoint helpers.

The architecture name is stored INSIDE every checkpoint, so a report can never
say "MobileNetV2" for weights that are really ResNet-18.
"""
import torch
import torch.nn as nn
from torchvision import models

SUPPORTED = ("resnet18", "mobilenet_v2")


def get_base_model(arch: str = "resnet18", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from {SUPPORTED}")
    return model


def save_checkpoint(path, model, arch, classes, split=None, seed=None, extra=None):
    """Save weights together with arch, class names and the exact data split used."""
    torch.save(
        {
            "arch": arch,
            "classes": list(classes),
            "state_dict": model.state_dict(),
            "split": split,   # {"train": [paths], "val": [...], "test": [...]}
            "seed": seed,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Rebuild the correct architecture from the checkpoint itself. Returns (model, ckpt)."""
    ckpt = torch.load(path, map_location=device)
    model = get_base_model(ckpt["arch"], num_classes=len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt