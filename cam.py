import os
import torch
import argparse
from torch import nn
from torch import Tensor
import torchvision
from torchvision.transforms import functional
from utils.torchcam.utils import overlay_mask
from utils.torchcam.cams import SmoothGradCAMpp
import matplotlib.pyplot as plt


import models
model_zoo = {
    'resnet34': models.Resnet34,
    'resnet34_nonlocal_layer1': models.Resnet34_NonLocal_layer1,
    'resnet34_nonlocal_layer2': models.Resnet34_NonLocal_layer2,
    'resnet34_nonlocal_layer3': models.Resnet34_NonLocal_layer3,
    'resnet34_nonlocal_layer4': models.Resnet34_NonLocal_layer4,
    'resnet50': models.Resnet50,
    'resnet50_nonlocal_layer1': models.Resnet50_NonLocal_layer1,
    'resnet50_nonlocal_layer2': models.Resnet50_NonLocal_layer2,
    'resnet50_nonlocal_layer3': models.Resnet50_NonLocal_layer3,
    'resnet50_nonlocal_layer4': models.Resnet50_NonLocal_layer4,
    'resnet50_nonlocal_5block': models.Resnet50_NonLocal_5block,
    'resnet50_nonlocal_10block': models.Resnet50_NonLocal_10block,
}
# --------------------------------------------------------
#   Args
# --------------------------------------------------------
parser = argparse.ArgumentParser(description='train')

parser.add_argument('--arch', type=str, default='resnet50', choices=model_zoo)
parser.add_argument('--checkpoint', type=str,
                    default='checkpoints/resnet50.pt')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='data/custom',
                    help='dataset folder')
parser.add_argument('--result_save_dir',  type=str, default='images')
args = parser.parse_args()

ARCH: int = args.arch
CHECKPOINT: str = args.checkpoint
DEVICE: str = args.device
DATASET_DIR: str = args.data  # 'data'
RESULT_DIR: str = args.result_save_dir


num_class = len(os.listdir(os.path.join(DATASET_DIR, 'test')))

model:nn.Module=model_zoo[ARCH]()
model.fc = nn.Linear(model.fc.in_features, num_class)
model.load_state_dict(torch.load(CHECKPOINT))
model.to(DEVICE)
model.eval()
cam_extractor = SmoothGradCAMpp(model)


for class_id, class_name in enumerate(os.listdir(os.path.join(DATASET_DIR, 'test'))):
    for file_id, file in enumerate(os.listdir(os.path.join(DATASET_DIR, 'test', class_name))[100:105]):
        full_file_path = os.path.join(DATASET_DIR, 'test', class_name, file)

        image = torchvision.io.image.read_image(full_file_path)
        input_tensor = functional.normalize(
            tensor=functional.resize(image, (224, 224)) / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        output_tensor: Tensor = model(input_tensor.unsqueeze(0).to(DEVICE))
        activation_map = cam_extractor(
            output_tensor.squeeze(0).argmax().item(),
            output_tensor)
        result = overlay_mask(
            img=functional.to_pil_image(image),
            mask=functional.to_pil_image(activation_map, mode='F'),
            alpha=0.5)

        plt.imshow(result)
        plt.axis('off')

        plt.tight_layout()
        # plt.show()

        os.makedirs(os.path.join(RESULT_DIR, ARCH, class_name), exist_ok=True)

        save_path = os.path.join(
            RESULT_DIR, ARCH, class_name, '%s.png' % (os.path.splitext(file)[0]))
        plt.savefig(save_path)

        print('class:%2d image:%2d %s save to %s' %
              (class_id, file_id, full_file_path, save_path))
