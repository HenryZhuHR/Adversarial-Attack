import os
import torch
from torch import nn
from torch import Tensor
import torchvision
from torchvision import models
from torchvision.transforms import functional
from torchcam.utils import overlay_mask
from torchcam.cams import SmoothGradCAMpp
import matplotlib.pyplot as plt
from models import ResNet_Attention


DEVICE = 'cuda'
DATASET_DIR = 'data'

RESULT_DIR = 'images/result'


# ----------------------------------------
#   Load dataset
# ----------------------------------------

num_class = len(os.listdir(os.path.join(DATASET_DIR, 'test')))
# ----------------------------------------
#   Load model and CAM
# ----------------------------------------
CHECKPOINT: str = 'checkpoints/noAttack.pt'
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_class)
model.load_state_dict(torch.load(CHECKPOINT))
model.to(DEVICE)
model.eval()
cam_extractor = SmoothGradCAMpp(model)

CHECKPOINT: str = 'checkpoints/resnet34_attention.pt'
model_attention = ResNet_Attention(num_class=num_class)
model_attention.fc = nn.Linear(model_attention.fc.in_features, num_class)
model_attention.load_state_dict(torch.load(CHECKPOINT))
model_attention.to(DEVICE)
model_attention.eval()
cam_extractor_attention = SmoothGradCAMpp(model_attention)


for class_id, class_name in enumerate(os.listdir(os.path.join(DATASET_DIR, 'test'))):
    for file_id, file in enumerate(os.listdir(os.path.join(DATASET_DIR, 'test', class_name))):
        full_file_path = os.path.join(DATASET_DIR, 'test', class_name, file)
        

        image = torchvision.io.image.read_image(full_file_path)
        input_tensor = functional.normalize(
            tensor=functional.resize(image, (224, 224)) / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        output_tensor:Tensor=model(input_tensor.unsqueeze(0).to(DEVICE))
        activation_map = cam_extractor(
            output_tensor.squeeze(0).argmax().item(), 
            output_tensor)
        result = overlay_mask(
            img=functional.to_pil_image(image), 
            mask=functional.to_pil_image(activation_map, mode='F'), 
            alpha=0.5)
        
        plt.subplot(2,1,1)
        plt.imshow(result); 
        plt.axis('off'); 
        plt.subplot(2,1,2)
        plt.imshow(result); 
        plt.axis('off'); 
        
        plt.tight_layout(); 
        # plt.show()

        os.makedirs(os.path.join(RESULT_DIR,class_name), exist_ok=True)

        save_path=os.path.join(RESULT_DIR,class_name,'%s.png'%(os.path.splitext(file)[0]))
        plt.savefig(save_path)

        print('class:%2d image:%2d %s save to %s' % (class_id, file_id, full_file_path,save_path))