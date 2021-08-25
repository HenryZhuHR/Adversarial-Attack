from .danet import ResNet_Attention

# === ResNetX with non-local adding in different stage ===
from .resnet_nonlocal import (
    Resnet34,
    Resnet34_NonLocal_layer1,
    Resnet34_NonLocal_layer2,
    Resnet34_NonLocal_layer3,
    Resnet34_NonLocal_layer4,
    Resnet50,
    Resnet50_NonLocal_layer1,
    Resnet50_NonLocal_layer2,
    Resnet50_NonLocal_layer3,
    Resnet50_NonLocal_layer4,
)