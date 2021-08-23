
from typing import Union
import numpy as np
import torch
from torch import nn
from torch import optim
from torch import Tensor



def fgsm(model, image: Tensor, label: Tensor,
         epsilon=0.01,
         clip_min: float = -1.0, clip_max: float = 1.0):
    device = image.device
    imageArray = image.detach().cpu().numpy()
    X_fgsm = Tensor(imageArray).to(device)

    X_fgsm.requires_grad = True

    optimizer = optim.SGD([X_fgsm], lr=1e-3)
    optimizer.zero_grad()

    loss_function = nn.CrossEntropyLoss()
    loss: Tensor = loss_function(model(X_fgsm), label)
    loss.backward()

    d = epsilon * X_fgsm.grad.data.sign()

    # 生成对抗样本
    x_adv = X_fgsm + d

    if clip_max == None and clip_min == None:
        clip_max = np.inf
        clip_min = -np.inf

    x_adv = torch.clamp(x_adv, clip_min, clip_max)
    x_adv.to(device)


    return x_adv
