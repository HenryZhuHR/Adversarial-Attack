import os
import copy
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


# --------------------------------------------------------
#   Args
# --------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='train')

# train parameter
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_worker', type=int, default=8)
parser.add_argument('--model_save_dir', type=str, default='checkpoints')
parser.add_argument('--model_save_name', type=str, default='resnet34')
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--logdir', type=str, default='runs')
args = parser.parse_args()

# train parameter
BATCH_SIZE: int = args.batch_size    # 128
NUM_WORKERS: int = args.num_worker   # 8
MODEL_SAVE_DIR: str = args.model_save_dir  # 'checkpoints'
MODEL_SAVE_NAME: str = args.model_save_name  # 'noAttack'
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
    # ----------------------------------------
    #   Load data
    # ----------------------------------------
    test_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'test'),
                                    transform=DATA_TRANSFORM['test'])
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS)
    
    num_class = len(test_set.classes)
    # ----------------------------------------
    #   Load model and fine tune
    # ----------------------------------------
    from torchvision import models
    model = models.resnet34(pretrained=True)
    # from models import ResNet_Attention
    # model=ResNet_Attention(num_class=num_class)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    model.to(DEVICE)

    running__acc=0.0
    num_data = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(test_loader)
        for images, labels in pbar:
            images: Tensor = images.to(DEVICE)
            labels: Tensor = labels.to(DEVICE)
            batch = images.size(0)
            num_data += batch

            output: Tensor = model(images)
            _, pred = torch.max(output, 1)
            epoch__acc = torch.sum(pred == labels).item()
            running__acc += epoch__acc
        valid_acc = running__acc / num_data