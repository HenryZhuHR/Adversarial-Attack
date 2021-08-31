import torch
import numpy as np
from torch import nn
from torch import optim
from torch import Tensor
from .base_attack import BaseAttack

class PGD(BaseAttack):
    attack_name = 'PGD'
    ATTACK_PARAMETERS = [
        'epsilon',
        'alpha',
        'num_steps'
    ]

    def __init__(self,
                 model,
                 device='cpu',
                 **kwargs  # {key:word} like attack argurement
                 ) -> None:
        super().__init__(model, device=device, **kwargs)
        self.attack_params = super().parse_params(kwargs=kwargs)
        # Init Attack Parameters
        self.epsilon=self.attack_params['epsilon']
        self.alpha=self.attack_params['alpha']
        self.num_steps=self.attack_params['num_steps']

    def attack(self, input: Tensor, label: Tensor,
               clip_min: float = -1.0, clip_max: float = 1.0) -> Tensor:
        X=Tensor(input.detach().cpu().numpy()).to(self.device)
        x_adv=X
        x_adv.requires_grad = True

        loss_function = nn.CrossEntropyLoss()

        for i in range(self.num_steps):
            outputs = self.model(x_adv)
            self.model.zero_grad()
            loss: Tensor = loss_function(outputs, label)
            loss.backward()

            # compute
            eta = self.alpha*x_adv.grad.detach().sign()
            x_adv = X+eta
            eta = torch.clamp(x_adv.detach()-X.detach(), min=-self.epsilon, max=self.epsilon)
            x_adv = torch.clamp(X+eta, min=0, max=1).detach()
            x_adv.requires_grad_()
            x_adv.retain_grad()

        return x_adv

