import os
from torch import nn

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
    Resnet50_NonLocal_5block,
    Resnet50_NonLocal_10block,
)

model_zoo = {
    'resnet34': Resnet34,
    'resnet34_nonlocal_layer1': Resnet34_NonLocal_layer1,
    'resnet34_nonlocal_layer2': Resnet34_NonLocal_layer2,
    'resnet34_nonlocal_layer3': Resnet34_NonLocal_layer3,
    'resnet34_nonlocal_layer4': Resnet34_NonLocal_layer4,
    'resnet50': Resnet50,
    'resnet50_nonlocal_layer1': Resnet50_NonLocal_layer1,
    'resnet50_nonlocal_layer2': Resnet50_NonLocal_layer2,
    'resnet50_nonlocal_layer3': Resnet50_NonLocal_layer3,
    'resnet50_nonlocal_layer4': Resnet50_NonLocal_layer4,
    'resnet50_nonlocal_5block': Resnet50_NonLocal_5block,
    'resnet50_nonlocal_10block': Resnet50_NonLocal_10block,
}


def GetModelByName(model_name: str) -> nn.Module:
    try:
        model = model_zoo[model_name]
        return model
    except KeyError as e:
        print('\033[31m[ERROR] No such model name: %s in model zoo:%s %s\033[0m' % (
            e, os.linesep, list(model_zoo.keys())))
        exit()
