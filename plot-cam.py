
import os
import cv2
import matplotlib.pyplot as plt

IMAGES_DIR = 'images'

model_list = [
    'resnet50',
    'resnet50_nonlocal_layer1',
    'resnet50_nonlocal_layer2',
    'resnet50_nonlocal_layer3',
    'resnet50_nonlocal_layer4',
    'resnet50_nonlocal_5block',
    'resnet50_nonlocal_10block'
]
class_list = os.listdir(os.path.join(IMAGES_DIR, model_list[0]))
file_list = os.listdir(os.path.join(IMAGES_DIR, model_list[0], class_list[0]))

N_ROW = len(model_list)
N_COL = len(file_list)
FIG_SIZE = (15, 15)
FIG_SIZE_W=FIG_SIZE[1]
FIG_SIZE_H=FIG_SIZE[0]
DPI = 50

print(N_ROW, N_COL)

for i_class, class_ in enumerate(class_list):
    # draw one pic
    figure_sub=plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    for i_model, model in enumerate(model_list):
        file_list = os.listdir(os.path.join(IMAGES_DIR, model, class_))
        for i_file, file in enumerate(file_list):
            file_path = os.path.join(IMAGES_DIR, model, class_, file)
            # print((i_model, i_file), file_path)

            plt.subplot(N_ROW, N_COL, (i_model*N_COL+i_file+1))
            image=cv2.imread(file_path)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            h,w,c=image.shape # 480 640 3
            # print(h,w,c)
            image=image[80:400,80:560] # (上：下，左：右)
            plt.imshow(image)
            plt.axis('off')
            if i_model==0:
                plt.title('%s-image-%d'%(class_,i_file),fontsize=20)
            if i_file==0:
                plt.text(-220,200,model.replace('_','\n'),fontsize=20)
            
    plt.tight_layout()
    plt.subplots_adjust(wspace=0,left=0.085)

    save_dir = 'result'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '%s.png' % (class_))
    plt.savefig(save_path)
    print('save figure to %s'%save_path)

    # plt.show()
    # exit()
