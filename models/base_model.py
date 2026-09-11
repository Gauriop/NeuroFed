import torch
import torch.nn as nn
import torchvision.models as models


def get_base_model(num_classes=4, pretrained=True):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model