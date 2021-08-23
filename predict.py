import torch
from torch import Tensor
from models.resnet_nonlocal import Resnet34_NonLocal_layer1
from models.resnet_nonlocal import Resnet34_NonLocal_layer2
from models.resnet_nonlocal import Resnet34_NonLocal_layer3
from models.resnet_nonlocal import Resnet34_NonLocal_layer4

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_size', type=int, default=224)
args = parser.parse_args()
IMAGE_SIZE: int = args.img_size


if __name__ == '__main__':
    for Model in [
        Resnet34_NonLocal_layer1,
        Resnet34_NonLocal_layer2,
        Resnet34_NonLocal_layer3,
        Resnet34_NonLocal_layer4
    ]:
        model = Model()
        x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        y: Tensor = model(x)
        print('model output==>', y.size())
