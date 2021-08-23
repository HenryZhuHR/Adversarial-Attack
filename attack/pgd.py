import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.cuda import device


def pgd_attack(
        model: nn.Module,
        X: Tensor,  # model input (image batch)
        y: Tensor,  # model output (label batch)
        alpha: float = 4. / 255,  # parameter of PGD attack
        epsilon: float = 0.3,  # parameter of PGD attack
        num_steps: int = 40,  # x+S item
        **kwargs) -> Tensor:
    device = X.device
    images = X.to(device)
    labels = y.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(num_steps):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def my_pgd_attack(
        model: nn.Module,
        X: Tensor,  # model input (image batch)
        y: Tensor,  # model output (label batch)
        alpha: float = 2. / 255,  # parameter of PGD attack
        epsilon: float = 8. / 255,  # parameter of PGD attack
        num_steps: int = 7,  # x+S item
        **kwargs) -> Tensor:
    """
        parameter
        ---
        - `model`
        - `x` model input (image batch)
        - `y` model output (label batch)

        FGSM: x'= x + epsilon * sign(\g_x J(x,y))
    """
    device: str = X.device
    # X_array=X.detach().cpu().numpy()
    # X_random=np.random.uniform(-epsilon,epsilon,X.shape)
    # X_array=np.clip(X_array+X_random,a_min=0.0,a_max=1.0)

    # x_adv=Tensor(X_array).to(device).float()
    x_adv = X
    x_adv.requires_grad = True

    loss_function = nn.CrossEntropyLoss().to(device)

    for i in range(num_steps):
        outputs = model(x_adv)
        model.zero_grad()
        loss: Tensor = loss_function(outputs, y)
        loss.backward()

        # compute
        eta = alpha*x_adv.grad.detach().sign()
        x_adv = X+eta
        eta = torch.clamp(x_adv.detach()-X.detach(), min=-epsilon, max=epsilon)
        x_adv = torch.clamp(X+eta, min=0, max=1).detach()
        x_adv.requires_grad_()
        x_adv.retain_grad()
        # print(eta.size())
        # exit()

    return x_adv
