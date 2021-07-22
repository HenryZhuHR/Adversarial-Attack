# 对抗样本的训练代码
## 基础代码
- [基本的训练代码](train.py)，基于resnet

## FGSM
[FGSM](train-fgsm.py)
```bash
python3 train-fgsm.py
```
backbone: resnet

FGSM(Fast Gradient Sign Method)，快速梯度下降法

在白盒环境下，通过求出模型对输入的导数，然后用符号函数得到其具体的梯度方向，接着乘以一个步长，得到的“扰动”加在原来的输入 上就得到了在FGSM攻击下的样本。
$$
x^{'}=x+\varepsilon \cdot sign(\triangledown_x J(x, y))
$$

## FGM
[FGM](train-fgm.py)，基于resnet
```bash
python3 train-fgm.py
```
## PGD
[PGD](train-pgd.py)
```bash
python3 train-pgd.py
```

backbone: resnet

git config --global user.name "HenryZhuHR"
git config --global user.email "296506195@qq.com"