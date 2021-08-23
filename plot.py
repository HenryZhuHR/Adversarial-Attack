import os
from pprint import pp
import numpy as np
from matplotlib import pyplot as plt

# plot size
FIG_SIZE = (9, 4.5)
FIG_SIZE_W=6
FIG_SIZE_H=8
DPI = 100
# subplot
LINE_COLOR: str = 'red'
LINE_ALPHA: float = 1.0
LINE_MARKER: str = '.'
X_LABEL_STR: str = 'epsilon'
X_LABEL_FONTSIZE: int = 13
Y_LABEL_FONTSIZE: int = 13
TITLE_FONTSIZE: int = 15
SET_GRID: bool = True  # 生成网格


def plot_fgsm(file: str, attack_method: str):
    with open(file, 'r')as f:
        data_list = []
        filecontent = f.readlines()[1:]
        for item in filecontent:
            item_list = item.rstrip().split('\t')
            data_list.append([
                int(item_list[0].split('/')[0]),  # e(int)
                float(item_list[0].split('=')[1]),  # epsilon
                float(item_list[1]),  # noAt acc
                float(item_list[2]),  # attack acc
                float(item_list[3]),  # noAt loss
                float(item_list[4]),  # attack loss
            ])
    import pprint
    pprint.pprint(data_list)
    print()
    e = np.array([d for d in [item[0] for item in data_list]])
    epsilon = np.array([d for d in [item[1] for item in data_list]])
    noAt_acc = np.array([d for d in [item[2] for item in data_list]])
    attack_acc = np.array([d for d in [item[3] for item in data_list]])
    noAt_loss = np.array([d for d in [item[4] for item in data_list]])
    attack_loss = np.array([d for d in [item[5] for item in data_list]])

    data_dict = {
        # 'noAttack accuracy':noAt_acc,
        'attack accuracy': attack_acc,
        # 'noAttack loss':noAt_loss,
        'attack loss': attack_loss,
    }

    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    for i, (name, data) in enumerate(data_dict.items()):
        set_type = name.split(' ')[0]
        data_type = name.split(' ')[1]
        plt.subplot(1, len(data_dict.items()), i+1)

        plt.plot(epsilon, data,  # 上数据
                 color=LINE_COLOR, alpha=LINE_ALPHA,
                 marker=LINE_MARKER)
        plt.xlabel(X_LABEL_STR, fontsize=X_LABEL_FONTSIZE)
        plt.ylabel(data_type, fontsize=Y_LABEL_FONTSIZE)
        # plt.xlim(0,100)
        # if data_type=='accuracy':
        #     plt.ylim(0,1)
        plt.title('%s %s %s' % (attack_method.upper(),
                  set_type, data_type), fontsize=TITLE_FONTSIZE)
        if SET_GRID:
            plt.grid()  # 生成网格

    if not os.path.exists('images/figures'):
        os.makedirs('images/figures')
    savefig_name = 'images/figures/%s.png' % attack_method

    # plt.subplots_adjust(wspace=0.35)
    plt.tight_layout()

    plt.savefig(savefig_name)
    # plt.show()
    plt.close()


def plot_pgd(file: str, attack_method: str):
    with open(file, 'r')as f:
        data_list = []
        filecontent = f.readlines()[1:]
        for item in filecontent:
            item_list = item.rstrip().split('\t')
            data_list.append([
                int(item_list[0].split('/')[0]),  # e(int)
                float(item_list[0].split('=')[-1]),  # epsilon
                int(item_list[1].split('/')[0]),  # a(int)
                float(item_list[1].split('=')[-1]),  # alpha
                int(item_list[2]),  # steps
                float(item_list[3]),  # noAt acc
                float(item_list[4]),  # attack acc
                float(item_list[5]),  # noAt loss
                float(item_list[6]),  # attack loss
            ])
    PGD_STEP_LIST = (10, 20, 40)
    PGD_E_LIST = (16, 64)               # e/255
    PGD_A_LIST = (1, 2, 4, 8, 16, 64)   # a/255

    # import pprint
    # pprint.pprint(data_list)
    plt.figure(num=0, figsize=(FIG_SIZE_W,FIG_SIZE_H), dpi=DPI)
    for i_step, step in enumerate(PGD_STEP_LIST):
        for i_e, e in enumerate(PGD_E_LIST):
            x = np.array([line[2]  # line[2]:a  line[0]:e  line[4]:step
                          for line in data_list if line[0] == e and line[4] == step], dtype=np.int8)
            y = np.array([line[6]  # line[6]:attack_acc  line[0]:e  line[4]:step
                          for line in data_list if line[0] == e and line[4] == step], dtype=np.float64)
            plot_index = int((i_e+1)+i_step*len(PGD_E_LIST))
            plt.subplot(len(PGD_STEP_LIST), len(PGD_E_LIST), plot_index)
            plt.plot(x, y,  # 画数据到子图上
                     color=LINE_COLOR, alpha=LINE_ALPHA,
                     marker=LINE_MARKER)
            plt.ylim(0,0.7)
            plt.title('e=%.4f,step=%d' % (e/255,step))
            if SET_GRID:
                    plt.grid()  # 生成网格
    if not os.path.exists('images/figures'):
        os.makedirs('images/figures')
    savefig_name = 'images/figures/pgd.png'
    plt.tight_layout()
    plt.savefig(savefig_name)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    DIR = 'logs'
    for file in os.listdir(DIR):
        if os.path.splitext(file)[-1] not in ['.txt']:
            continue
        attack_method = os.path.splitext(os.path.split(file)[-1])[0]
        if attack_method in ['fgm', 'fgsm']:
            plot_fgsm(os.path.join(DIR, file), attack_method)
        if attack_method in ['pgd']:
            plot_pgd(os.path.join(DIR, file), attack_method)
