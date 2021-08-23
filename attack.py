from attack.pgd import pgd_attack, my_pgd_attack
import os
import cv2
import tqdm
import numpy as np
import torch
from torch import nn
from torch import cuda
from torch import optim
from torch import Tensor
from torch.utils import data
import torchvision
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


# --------------------------------------------------------
#   Args
# --------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--attack', type=str, default='fgsm', help='attack method',
                    choices=['fgsm', 'fgm', 'pgd'])
# attack parameter
parser.add_argument('--epsilon', type=float, default=2)
parser.add_argument('--alpha', type=float, default=2)
parser.add_argument('--num_steps', type=int, default=40)
# train parameter
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_worker', type=int, default=8)
parser.add_argument('--logs', type=str, default='logs')
parser.add_argument('--pretrained', type=str,
                    default='checkpoints/noAttack.pt')
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--logdir', type=str, default='runs')
args = parser.parse_args()

# attack parameter
EPSILON: float = args.epsilon/255.
ALPHA: float = args.alpha/255.
NUM_STEPS: int = args.num_steps
# train parameter
BATCH_SIZE: int = args.batch_size    # 1
NUM_WORKERS: int = args.num_worker   # 8
LOGS: str = args.logs  # 'logs'
PRETRAINED: str = args.pretrained
DATASET_DIR: str = args.data  # 'data'
LOG_DIR: str = args.logdir    # 'runs'
DEVICE: str = args.device

DATA_TRANSFORM = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                 # transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()]),  # 来自官网参数
    'valid': transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()]),
    'test': transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
}


if __name__ == '__main__':

    writer = SummaryWriter('runs/pgd')

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
    #   Load model and fine tune
    # ----------------------------------------
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    if not os.path.exists(PRETRAINED):
        raise FileExistsError(
            "No such .pt file: %s, please run 'python3 attack-noAttack.py'" % PRETRAINED)
    model.load_state_dict(torch.load(PRETRAINED))
    model.to(DEVICE)
    loss_function = nn.CrossEntropyLoss().to(DEVICE)

    print("\033[0;32;40m[PGD]\033[0m")
    # no Attack
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

        output: Tensor = model(images)
        _, pred = torch.max(output, 1)
        loss: Tensor = loss_function(output, labels)

        epoch_loss = loss.item()
        epoch__acc = torch.sum(pred == labels).item()
        running_loss += epoch_loss
        running__acc += epoch__acc

        pbar_show = (epoch_loss / batch, epoch__acc / batch)
        pbar.set_description(' noAttack loss:%.6f  acc:%.6f' % pbar_show)
    noAttack_loss = running_loss / num_data
    noAttack_acc = running__acc / num_data

    # Strat Attack
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

        # ------------------------------
        # ------------------------------
        images_adv = pgd_attack(model, images, labels,
                                alpha=ALPHA, epsilon=EPSILON, num_steps=NUM_STEPS)
        # ------------------------------
        if step <= 1:
            show_img = np.min([batch-1, 8-1])
            writer.add_images(
                'image/src', images[0:show_img], global_step=step)
            writer.add_images(
                'image/adv', images_adv[0:show_img], global_step=step)
            writer.add_images(
                'image/perturbation', images_adv[0:show_img]-images[0:show_img], global_step=step)
            if not os.path.exists('images/pgd'):
                os.makedirs('images/pgd')
            from attack.utils import save_image_tensor
            save_image_tensor('images/pgd/e=%d~a=%d~s=%d-src.png' %
                              (int(EPSILON*255), int(ALPHA*255), NUM_STEPS), images[0:show_img])
            save_image_tensor('images/pgd/e=%d~a=%d~s=%d-adv.png' %
                              (int(EPSILON*255), int(ALPHA*255), NUM_STEPS), images_adv[0:show_img])
            save_image_tensor('images/pgd/e=%d~a=%d~s=%d-per.png' % (int(EPSILON*255), int(
                ALPHA*255), NUM_STEPS), images_adv[0:show_img]-images[0:show_img])

        output: Tensor = model(images_adv)
        _, pred = torch.max(output, 1)
        loss: Tensor = loss_function(output, labels)

        epoch_loss = loss.item()
        epoch__acc = torch.sum(pred == labels).item()
        running_loss += epoch_loss
        running__acc += epoch__acc

        pbar_show = (epoch_loss / batch, epoch__acc / batch)
        pbar.set_description(' Attack loss:%.6f  acc:%.6f' % pbar_show)
    attack_loss = running_loss / num_data
    attack_acc = running__acc / num_data

    hparam_dict = {'epsilon': EPSILON, 'alpha': ALPHA, 'num steps': NUM_STEPS}
    metric_dict = {
        'noAttack accuracy': noAttack_acc, 'attack accuracy': attack_acc,
        'noAttack loss': noAttack_loss, 'attack loss': attack_loss
    }
    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()

    if not os.path.exists(LOGS):
        os.makedirs(LOGS)

    if not os.path.exists(os.path.join(LOGS, 'pgd.txt')):
        with open(os.path.join(LOGS, 'pgd.txt'), 'a') as f:
            f.write(
                '(e/255)epsilon\t(a/255)alpha\ts\tnoAt acc\tattack acc\tnoAt loss\tattack loss\n')
    with open(os.path.join(LOGS, 'pgd.txt'), 'a') as f:
        # f.write('%d/255=%f\t%f\t%f\t%d\t%f\t%f\t%f\n'%(int(EPSILON*255),EPSILON,ALPHA,NUM_STEPS,noAttack_acc,attack_acc,noAttack_loss,attack_loss))
        f.write('{e}/255={epsilon:.6f}\t{a}/255={alpha:.6f}\t{num_steps}\t{noAttack_acc:.6f}\t{attack_acc:.6f}\t{noAttack_loss:.6f}\t{attack_loss:.6f}\n'.format(
            e=int(EPSILON*255),
            epsilon=EPSILON,
            a=int(ALPHA*255),
            alpha=ALPHA,
            num_steps=NUM_STEPS,
            noAttack_acc=noAttack_acc,
            attack_acc=attack_acc,
            noAttack_loss=noAttack_loss,
            attack_loss=attack_loss
        ))
