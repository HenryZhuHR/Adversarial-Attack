
from ctypes import DEFAULT_MODE
import os
import copy
import tqdm
import argparse
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
from torch.utils.tensorboard import SummaryWriter

from modules import attack
from modules import models as Models
# --------------------------------------------------------
#   Args
# --------------------------------------------------------
parser = argparse.ArgumentParser(description='train')

# train parameter
parser.add_argument('--arch', type=str, choices=Models.model_zoo,
                    default=list(Models.model_zoo.keys())[0],
                    # default='resnet50',
                    )
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_worker', type=int, default=8)
parser.add_argument('--model_save_dir', type=str, default='server/checkpoints')
parser.add_argument('--model_save_name', type=str,
                    help='using arch name if not given')
parser.add_argument('--data', type=str, default='%s/datasets/custom' % os.path.expanduser('~'),
                    help='dataset folder')
parser.add_argument('--logdir', type=str, default='server/runs',
                    help='train log save folder')
parser.add_argument('--model_summary', action='store_true',
                    help='if print model summary')
# adversarial training
parser.add_argument('--adversarial_training', action='store_true',
                    help='if adversarial training')
parser.add_argument('--attack_method', type=str, default='fgsm')
parser.add_argument('--epsilon', type=float, default=0)
args = parser.parse_args()

# Parse Args
ARCH: int = args.arch
DEVICE: str = args.device
BATCH_SIZE: int = args.batch_size    # 128
MAX_EPOCH: int = args.max_epoch    # 100
LR: float = args.lr   # 0.01
NUM_WORKERS: int = args.num_worker   # 8
MODEL_SAVE_DIR: str = args.model_save_dir  # 'checkpoints'
MODEL_SAVE_NAME: str = ARCH if args.model_save_name == None else args.model_save_name  # 'NONE'/ARCH
DATASET_DIR: str = args.data  # 'data'
LOG_DIR: str = args.logdir    # 'where tensorboard data save (runs)'
IS_MODEL_SUMMARY: bool = args.model_summary
# adversarial training
IS_ADVERSARIAL_TRAINING: bool = args.adversarial_training
ATTACK_METHOD: str = args.attack_method
EPSILON: float = args.epsilon/255.
ATTACK_PARAM_dict = {
    'epsilon': EPSILON,
    'alpha': EPSILON,
}

DATA_TRANSFORM = {
    'train': transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor()]),
    'valid': transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()]),
    'test': transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()])
}

if __name__ == '__main__':
    os.makedirs('%s/%s' % (LOG_DIR, ARCH), exist_ok=True)

    writer = SummaryWriter('%s/%s' % (LOG_DIR, ARCH))

    # ----------------------------------------
    #   Load dataset
    # ----------------------------------------
    if ARCH == 'mnist_cnn':  # https://pytorch.org/vision/stable/datasets.html#mnist
        train_set = torchvision.datasets.MNIST(root=os.path.join(os.path.expanduser('~'), 'datasets'),
                                               train=True, download=True,transform=transforms.ToTensor())
        valid_set = torchvision.datasets.MNIST(root=os.path.join(os.path.expanduser('~'), 'datasets'),
                                               train=False, download=True,transform=transforms.ToTensor())
        test_set = torchvision.datasets.MNIST(root=os.path.join(os.path.expanduser('~'), 'datasets'),
                                               train=False, download=True,transform=transforms.ToTensor())
    else:
        train_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'train'),
                                     transform=DATA_TRANSFORM['train'])
        valid_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'valid'),
                                        transform=DATA_TRANSFORM['valid'])
        test_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'test'),
                                        transform=DATA_TRANSFORM['test'])

    valid_loader = data.DataLoader(valid_set, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS)    
    test_loader = data.DataLoader(test_set, batch_size=1,
                                  shuffle=False, num_workers=NUM_WORKERS)
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=NUM_WORKERS)
    

    
    num_class = len(train_set.classes)
    # ----------------------------------------
    #   Load model and fine tune
    # ----------------------------------------
    print('Try to load model \033[0;32;40m%s\033[0m ...' % ARCH)
    model: nn.Module = Models.model_zoo[ARCH]()
    model.fc = nn.Linear(model.fc.in_features, num_class)
    # -------------------- fine tune --------------------
    # FINE_TUNE=True
    # if FINE_TUNE:
    #     model_state_dict=torch.load('checkpoints/resnet50-best.pt')
    #     for name, param in model.named_parameters():
    #         print(name,end='  ')
    #         if name in model_state_dict:
    #             print('--',end='')
    #             param=model_state_dict[name]
    #             param.requires_grad = False
    #         print()

    model.to(DEVICE)

    if IS_ADVERSARIAL_TRAINING:
        model_attack: attack.BaseAttack = attack.GetAttackByName(
            ATTACK_METHOD)(model, DEVICE, **ATTACK_PARAM_dict)

    # ----------------------------------------
    #   tensorboard :   Add model graph
    #   torchsummary:   Summary model
    # ----------------------------------------
    input_tensor_sample: Tensor = train_set[0][0]
    writer.add_graph(model, input_to_model=(
        input_tensor_sample.unsqueeze(0)).to(DEVICE))
    if IS_MODEL_SUMMARY:
        try:
            from torchsummary import summary
        except:
            print('please install torchsummary by command: pip instsll torchsummary')
        else:
            print(summary(model, input_tensor_sample.size(),
                  device=DEVICE.split(':')[0]))

    loss_function = nn.CrossEntropyLoss()
    loss_function.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ----------------------------------------
    #   Train model
    # ----------------------------------------
    print('train model in device: \033[0;32;40m%s\033[0m ' % DEVICE)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    train_log = []
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_valid_acc = 0.0
    for epoch in range(1, MAX_EPOCH + 1):
        print('\033[0;32;40m[train: %s]\033[0m' % ARCH, end=' ')
        print('[Epoch] %d/%d' % (epoch, MAX_EPOCH), end=' ')
        print('[Batch Size] %d' % (BATCH_SIZE), end=' ')
        print('[LR] %f' % (LR))

        # --- train ---
        running_loss, running__acc = 0.0, 0.0
        num_data = 0    # how many data has trained
        model.train()
        pbar = tqdm.tqdm(train_loader)
        for images, labels in pbar:
            images: Tensor = images.to(DEVICE)
            labels: Tensor = labels.to(DEVICE)
            batch = images.size(0)
            num_data += batch

            if IS_ADVERSARIAL_TRAINING:
                images = model_attack.attack(images, labels)

            output: Tensor = model(images)
            _, pred = torch.max(output, 1)
            loss: Tensor = loss_function(output, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss = loss.item()
            epoch__acc = torch.sum(pred == labels).item()
            running_loss += epoch_loss
            running__acc += epoch__acc

            pbar.set_description('loss:%.6f acc:%.6f' %
                                 (epoch_loss / batch, epoch__acc / batch))
        train_loss = running_loss / num_data
        train_acc = running__acc / num_data

        # --- valid ---
        running_loss, running__acc = 0.0, 0.0
        num_data = 0
        model.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(valid_loader)
            for images, labels in pbar:
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

                pbar.set_description('loss:%.6f acc:%.6f' %
                                     (epoch_loss / batch, epoch__acc / batch))
            valid_loss = running_loss / num_data
            valid_acc = running__acc / num_data

        print('Train Loss:%f Accuracy:%f' % (train_loss, train_acc))
        print('Valid Loss:%f Accuracy:%f' % (valid_loss, valid_acc))

        writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
        writer.add_scalar('Train/Accuracy', train_acc, global_step=epoch)
        writer.add_scalar('Valid/Loss', valid_loss, global_step=epoch)
        writer.add_scalar('Valid/Accuracy', valid_acc, global_step=epoch)

        if valid_acc > best_valid_acc:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_valid_acc = valid_acc
        torch.save(model.state_dict(), os.path.join(
            MODEL_SAVE_DIR, '%s.pt' % ARCH))

        train_log.append([epoch, train_loss, train_acc, valid_loss, valid_acc])

    # save the best model
    model.load_state_dict(best_model_state_dict)
    torch.save(model.state_dict(), os.path.join(
        MODEL_SAVE_DIR, '%s-best.pt' % ARCH))

    # ----------------------------------------
    #   Test model
    # ----------------------------------------
    print('\033[0;32;40m[Test: %s]\033[0m' % ARCH)
    running_loss, running__acc = 0.0, 0.0
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
            loss: Tensor = loss_function(output, labels)

            epoch_loss = loss.item()
            epoch__acc = torch.sum(pred == labels).item()
            running_loss += epoch_loss
            running__acc += epoch__acc

            pbar.set_description('loss:%.6f acc:%.6f' %
                                 (epoch_loss / batch, epoch__acc / batch))
        test_loss = running_loss / num_data
        test_acc = running__acc / num_data
        print('Test Loss:%f Accuracy:%f' % (test_loss, test_acc))

    # ----------------------------------------
    #   Write logs
    # ----------------------------------------
    hparam_dict = {'batch size': BATCH_SIZE, 'lr': LR}
    metric_dict = {
        'train loss': train_loss, 'train accuracy': train_acc,
        'valid loss': valid_loss, 'valid accuracy': valid_acc,
        'test accuracy': test_acc
    }
    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()

    os.makedirs('logs', exist_ok=True)
    import time
    finished_time = time.strftime("%m%d_%H%M", time.localtime())
    with open(os.path.join(LOG_DIR, '%s-%s.txt' % (ARCH, finished_time)), 'w') as f:
        f.write('bacth size =%d\n' % BATCH_SIZE)
        f.write('lr         =%f\n' % LR)
        f.write('train epoch=%d\n' % epoch)
        f.write('device     =%s\n' % DEVICE)
        f.write('test_loss:%f,test_acc:%f\n' % (test_loss, test_acc))
        f.write('epoch\ttrain loss\ttrain accuracy\tvalid loss\tvalid accuracy\n')
        for item in train_log:
            f.write('{epoch:.6f}\t{train_loss:.6f}\t{train_acc:.6f}\t{valid_loss:.6f}\t{valid_acc:.6f}\n'.format(
                epoch=epoch, train_loss=train_loss, train_acc=train_acc, valid_loss=valid_loss, valid_acc=valid_acc
            ))
