import os
from PIL import Image
import tqdm
import numpy as np
import torch
from torch import nn
from torch import cuda
from torch import optim
from torch import Tensor
from torch.utils import data
import torchvision
from torchvision import datasets
from torchvision import transforms
from modules import attack
from modules.models import model_zoo, GetModelByName

# --------------------------------------------------------
#   Args
# --------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--arch', type=str, choices=model_zoo,
                    # default=list(model_zoo.keys())[0]
                    default='resnet50'
                    )
parser.add_argument('--checkpoint', type=str,
                    default='checkpoints/resnet50-best.pt')
# attack parameters
parser.add_argument('--attack_method', type=str, default='fgsm')
parser.add_argument('--epsilon', type=float, default=0)
# other parameters
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--data', type=str, default='data/custom')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_worker', type=int, default=0)
parser.add_argument('--logs', type=str, default='server/logs')


parser.add_argument('--logdir', type=str, default='server/runs')
args = parser.parse_args()


ARCH: int = args.arch
CHECKPOINT: str = args.checkpoint
# attack parameters
ATTACK_METHOD: str = args.attack_method
EPSILON: float = args.epsilon/255.
ATTACK_PARAM_dict = {
    'epsilon': EPSILON,
    'alpha': EPSILON,
}
# other parameters
DEVICE: str = args.device
DATASET_DIR: str = args.data  # 'data'
BATCH_SIZE: int = args.batch_size    # 1
NUM_WORKERS: int = args.num_worker   # 8
LOGS: str = args.logs  # 'logs'
LOG_DIR: str = args.logdir    # 'runs'


DATA_TRANSFORM = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.ToTensor()]),  # 来自官网参数
    'valid': transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()]),
    'test': transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
}

if __name__ == '__main__':
    # ----------------------------------------
    #   Load data
    # ----------------------------------------
    train_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'train'),
                                     transform=DATA_TRANSFORM['train'])
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=NUM_WORKERS)

    valid_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'valid'),
                                     transform=DATA_TRANSFORM['valid'])
    valid_loader = data.DataLoader(valid_set, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS)
    test_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'test'),
                                    transform=DATA_TRANSFORM['test'])
    test_loader = data.DataLoader(test_set, batch_size=1,
                                  shuffle=False, num_workers=NUM_WORKERS)

    num_class = len(train_set.classes)

    # ----------------------------------------
    #   Load model and checkpoint
    # ----------------------------------------
    model: nn.Module = GetModelByName(ARCH)()
    model.fc = nn.Linear(model.fc.in_features, num_class)
    model.to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT))
    loss_function = nn.CrossEntropyLoss().to(DEVICE)

    model_attack: attack.BaseAttack = attack.GetAttackByName(
        ATTACK_METHOD)(model, DEVICE, **ATTACK_PARAM_dict)

    # Strat Attack
    print("\033[0;32;40m[%s attack]\033[0m" % (model_attack.attack_name))
    running_loss, running__acc = 0.0, 0.0
    num_data, step = 0, 0
    model.eval()
    pbar = tqdm.tqdm(valid_loader)
    for images, labels in pbar:
        step += 1
        images: Tensor = images.to(DEVICE)
        labels: Tensor = labels.to(DEVICE)
        batch = images.size(0)
        num_data += batch

        images_adv = model_attack.attack(images, labels)
        if step <= 1:
            show_img = np.min([batch-1, 8-1])
            if not os.path.exists('images/fgsm'):
                os.makedirs('images/fgsm')
            attack.save_image_tensor('images/fgsm/e=%d-src.png' %
                              (int(EPSILON*255)), images[0:show_img])
            attack.save_image_tensor('images/fgsm/e=%d-adv.png' %
                              (int(EPSILON*255)), images_adv[0:show_img])
            attack.save_image_tensor('images/fgsm/e=%d-per.png' %
                              (int(EPSILON*255)), images_adv[0:show_img]-images[0:show_img])

        output: Tensor = model(images_adv)
        _, pred = torch.max(output, 1)
        loss: Tensor = loss_function(output, labels)

        epoch_loss = loss.item()
        epoch__acc = torch.sum(pred == labels).item()
        running_loss += epoch_loss
        running__acc += epoch__acc

        pbar.set_description('[%d tensor/batch] Valid loss:%.6f  acc:%.6f' %
                             (batch, epoch_loss / batch, epoch__acc / batch))
    attack_loss = running_loss / num_data
    attack_acc = running__acc / num_data

    if not os.path.exists(LOGS):
        os.makedirs(LOGS)
    if not os.path.exists(os.path.join(LOGS, 'fgsm.txt')):
        with open(os.path.join(LOGS, 'fgsm.txt'), 'a') as f:
            f.write('(e/255)epsilon\tattack acc\tattack loss\n')
    with open(os.path.join(LOGS, 'fgsm.txt'), 'a') as f:
        f.write(
            '{epsilon_int}/255={epsilon:.6f}\t{attack_acc:.6f}\t{attack_loss:.6f}\n'.format(
                epsilon_int=int(EPSILON*255),
                epsilon=EPSILON,
                attack_acc=attack_acc,
                attack_loss=attack_loss
            )
        )
