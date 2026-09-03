import torch
import torch.nn as nn
import torchvision.models as models


def get_base_model(num_classes=4, pretrained=True):
    """
    Returns a ResNet-18 adapted for brain tumor classification.
    pretrained=True uses ImageNet weights (transfer learning).
    """
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_model_weights(model):
    """Extract model weights as a state_dict (used when clients 'send' updates)."""
    return {k: v.clone() for k, v in model.state_dict().items()}


def set_model_weights(model, weights):
    """Load a state_dict back into a model (used when server sends the global model)."""
    model.load_state_dict(weights)
    return model


if __name__ == "__main__":
    m = get_base_model()
    dummy_input = torch.randn(1, 3, 128, 128)
    output = m(dummy_input)
    print("Output shape:", output.shape)