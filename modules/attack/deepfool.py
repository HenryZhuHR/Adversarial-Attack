from typing import List
import torch
import numpy as np
from torch import nn
from torch import optim
from torch import Tensor
from .base_attack import BaseAttack



import numpy as np
from torch.autograd import Variable
import torch as torch
import copy

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()


class DeepFool(BaseAttack):
    """
    DeepFool attack
    ===

    Paper: 
    Code : 

    :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    def __init__(self,
                 model,
                 device='cpu',
                 attack_params: List[int] = []
                 ) -> None:
        super().__init__(model, device=device)
        self.attack_name = 'DeepFool'
        self.ATTACK_PARAMETERS.update(
            num_classes=int(10),  # num classes
            overshoot=float(0.02),
            max_iter=int(1000),

        )
        super().parse_params(attack_params)

        self.num_classes = int(self.ATTACK_PARAMETERS['num_classes'])
        self.overshoot = float(self.ATTACK_PARAMETERS['overshoot'])
        self.max_iter = int(self.ATTACK_PARAMETERS['max_iter'])

    def attack(self, images: Tensor, labels: Tensor) -> Tensor:
        f_image = self.model.forward(images).data.cpu().numpy().flatten()
        output = (np.array(f_image)).flatten().argsort()[::-1]

        output = output[0:self.num_classes]
        label = output[0]

        input_shape = images.cpu().numpy().shape
        x = copy.deepcopy(images).requires_grad_(True)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        fs = self.model.forward(x)
        # fs_list = [fs[0,output[k]] for k in range(self.num_classes)]
        current_pred_label = label

        for i in range(self.max_iter):

            pert = np.inf
            fs[0, output[0]].backward(retain_graph = True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, self.num_classes):
                zero_gradients(x)

                fs[0, output[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, output[k]] - fs[0, output[0]]).data.cpu().numpy()

                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i =  (pert+1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            pert_image = images + (1+self.overshoot)*torch.from_numpy(r_tot).to(self.device)

            x = pert_image.detach().requires_grad_(True)
            fs = self.model.forward(x)

            if (not np.argmax(fs.data.cpu().numpy().flatten()) == label):
                break


        r_tot = (1+self.overshoot)*r_tot

        return pert_image
