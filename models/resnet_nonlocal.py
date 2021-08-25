import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.linear import Linear
from torchvision.models.resnet import (
    resnet34,
    resnet50
)

# from .non_local.gaussian import NONLocalBlock2D
from .non_local.embedded_gaussian import NONLocalBlock2D
# from .non_local.dot_product import NONLocalBlock2D
# from .non_local.concatenation import NONLocalBlock2D


class Resnet34(nn.Module):
    def __init__(self, channel: int = 3, pretrained=True):
        super(Resnet34, self).__init__()
        self.backbone = resnet34(pretrained=pretrained)
        self.fc = Linear(self.backbone.fc.in_features,
                         self.backbone.fc.out_features)

    def forward(self, x: Tensor):
        batch_size, channel, height, width = x.size()  # NCHW
        assert height >= 224 and width >= 224
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # [1, 64,  56, 56]
        x = self.backbone.layer2(x)  # [1, 128, 28, 28]
        x = self.backbone.layer3(x)  # [1, 256, 14, 14]
        x = self.backbone.layer4(x)  # [1, 512,  7,  7]

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Resnet34_NonLocal_layer1(nn.Module):
    def __init__(self):
        super(Resnet34_NonLocal_layer1, self).__init__()
        self.backbone = resnet34(pretrained=True)
        self.non_local = NONLocalBlock2D(in_channels=64)
        self.fc = Linear(self.backbone.fc.in_features,
                         self.backbone.fc.out_features)

    def forward(self, x: Tensor):
        batch_size, channel, height, width = x.size()  # NCHW
        assert height >= 224 and width >= 224
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # [1, 64,  56, 56]
        x = self.non_local(x)
        x = self.backbone.layer2(x)  # [1, 128, 28, 28]
        x = self.backbone.layer3(x)  # [1, 256, 14, 14]
        x = self.backbone.layer4(x)  # [1, 512,  7,  7]

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Resnet34_NonLocal_layer2(nn.Module):
    def __init__(self):
        super(Resnet34_NonLocal_layer2, self).__init__()
        self.backbone = resnet34(pretrained=True)
        self.non_local = NONLocalBlock2D(in_channels=128)
        self.fc = Linear(self.backbone.fc.in_features,
                         self.backbone.fc.out_features)

    def forward(self, x: Tensor):
        batch_size, channel, height, width = x.size()  # NCHW
        assert height >= 224 and width >= 224
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # [1, 64,  56, 56]
        x = self.backbone.layer2(x)  # [1, 128, 28, 28]
        x = self.non_local(x)
        x = self.backbone.layer3(x)  # [1, 256, 14, 14]
        x = self.backbone.layer4(x)  # [1, 512,  7,  7]

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Resnet34_NonLocal_layer3(nn.Module):
    def __init__(self):
        super(Resnet34_NonLocal_layer3, self).__init__()
        self.backbone = resnet34(pretrained=True)
        self.non_local = NONLocalBlock2D(in_channels=256)
        self.fc = Linear(self.backbone.fc.in_features,
                         self.backbone.fc.out_features)

    def forward(self, x: Tensor):
        batch_size, channel, height, width = x.size()  # NCHW
        assert height >= 224 and width >= 224
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # [1, 64,  56, 56]
        x = self.backbone.layer2(x)  # [1, 128, 28, 28]
        x = self.backbone.layer3(x)  # [1, 256, 14, 14]
        x = self.non_local(x)
        x = self.backbone.layer4(x)  # [1, 512,  7,  7]

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Resnet34_NonLocal_layer4(nn.Module):
    def __init__(self):
        super(Resnet34_NonLocal_layer4, self).__init__()
        self.backbone = resnet34(pretrained=True)
        self.non_local = NONLocalBlock2D(in_channels=512)
        self.fc = Linear(self.backbone.fc.in_features,
                         self.backbone.fc.out_features)

    def forward(self, x: Tensor):
        batch_size, channel, height, width = x.size()  # NCHW
        assert height >= 224 and width >= 224
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # [1, 64,  56, 56]
        x = self.backbone.layer2(x)  # [1, 128, 28, 28]
        x = self.backbone.layer3(x)  # [1, 256, 14, 14]
        x = self.backbone.layer4(x)  # [1, 512,  7,  7]
        x = self.non_local(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Resnet50(nn.Module):
    def __init__(self, channel: int = 3, pretrained=True):
        super(Resnet50, self).__init__()
        self.backbone = resnet50(pretrained=pretrained)
        self.fc = Linear(self.backbone.fc.in_features,
                         self.backbone.fc.out_features)

    def forward(self, x: Tensor):
        batch_size, channel, height, width = x.size()  # NCHW
        assert height >= 224 and width >= 224
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # [1, 256, 56, 56]
        x = self.backbone.layer2(x)  # [1, 512, 28, 28]
        x = self.backbone.layer3(x)  # [1, 1024, 14, 14]
        x = self.backbone.layer4(x)  # [1, 2048,  7,  7]

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Resnet50_NonLocal_layer1(nn.Module):
    def __init__(self):
        super(Resnet50_NonLocal_layer1, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.non_local = NONLocalBlock2D(in_channels=256)
        self.fc = Linear(self.backbone.fc.in_features,
                         self.backbone.fc.out_features)

    def forward(self, x: Tensor):
        batch_size, channel, height, width = x.size()  # NCHW
        assert height >= 224 and width >= 224
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # [1, 256, 56, 56]
        x = self.non_local(x)
        x = self.backbone.layer2(x)  # [1, 512, 28, 28]
        x = self.backbone.layer3(x)  # [1, 1024, 14, 14]
        x = self.backbone.layer4(x)  # [1, 2048,  7,  7]

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Resnet50_NonLocal_layer2(nn.Module):
    def __init__(self):
        super(Resnet50_NonLocal_layer2, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.non_local = NONLocalBlock2D(in_channels=512)
        self.fc = Linear(self.backbone.fc.in_features,
                         self.backbone.fc.out_features)

    def forward(self, x: Tensor):
        batch_size, channel, height, width = x.size()  # NCHW
        assert height >= 224 and width >= 224
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # [1, 256, 56, 56]
        x = self.backbone.layer2(x)  # [1, 512, 28, 28]
        x = self.non_local(x)
        x = self.backbone.layer3(x)  # [1, 1024, 14, 14]
        x = self.backbone.layer4(x)  # [1, 2048,  7,  7]

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Resnet50_NonLocal_layer3(nn.Module):
    def __init__(self):
        super(Resnet50_NonLocal_layer3, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.non_local = NONLocalBlock2D(in_channels=1024)
        self.fc = Linear(self.backbone.fc.in_features,
                         self.backbone.fc.out_features)

    def forward(self, x: Tensor):
        batch_size, channel, height, width = x.size()  # NCHW
        assert height >= 224 and width >= 224
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # [1, 256, 56, 56]
        x = self.backbone.layer2(x)  # [1, 512, 28, 28]
        x = self.backbone.layer3(x)  # [1, 1024, 14, 14]
        x = self.non_local(x)
        x = self.backbone.layer4(x)  # [1, 2048,  7,  7]

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Resnet50_NonLocal_layer4(nn.Module):
    def __init__(self):
        super(Resnet50_NonLocal_layer4, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.non_local = NONLocalBlock2D(in_channels=2048)
        self.fc = Linear(self.backbone.fc.in_features,
                         self.backbone.fc.out_features)

    def forward(self, x: Tensor):
        batch_size, channel, height, width = x.size()  # NCHW
        assert height >= 224 and width >= 224
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  # [1, 256, 56, 56]
        x = self.backbone.layer2(x)  # [1, 512, 28, 28]
        x = self.backbone.layer3(x)  # [1, 1024, 14, 14]
        x = self.backbone.layer4(x)  # [1, 2048,  7,  7]
        x = self.non_local(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

