import torch.nn as nn
import torchvision.models as models


def get_base_model(arch="resnet18", num_classes=4, pretrained=True):
    if arch == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model