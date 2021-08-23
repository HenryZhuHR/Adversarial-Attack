# 对抗样本的训练代码
- [对抗样本的训练代码](#对抗样本的训练代码)
- [服务器文件传输](#服务器文件传输)
- [tmux](#tmux)
  - [新建与关闭](#新建与关闭)
  - [窗口](#窗口)
- [Tensorboard](#tensorboard)
- [CAM](#cam)
- [运行](#运行)
- [代码细节](#代码细节)
    - [基础代码](#基础代码)
    - [FGSM](#fgsm)
    - [FGM](#fgm)
    - [PGD](#pgd)
# 服务器文件传输
`scp` 命令
```bash
# upload local_file to remote_folder
scp local_file remote_username@remote_ip:remote_folder 
# upload local_file to remote named remote_file
scp local_file remote_username@remote_ip:remote_file 

# upload local_folder to remote_folder/
scp -r local_folder remote_username@remote_ip:remote_folder
```

- **本地** 上传目录
```ps1
scp -r `
  /Users/Henryzhu/project/deeplearning-cv/advers `
  ubuntu@192.168.101.11:/home/ubuntu/project
scp -r `
  /Users/Henryzhu/project/deeplearning-cv/advers/models `
  ubuntu@192.168.101.11:/home/ubuntu/project/advers
```
- **本地** 上传单个文件
```ps1
scp `
  /Users/Henryzhu/project/deeplearning-cv/advers/models/danet.py `
  ubuntu@192.168.101.11:/home/ubuntu/project/advers/models/danet.py
scp `
  /Users/Henryzhu/project/deeplearning-cv/advers/train-nonlocal.sh `
  ubuntu@192.168.101.11:/home/ubuntu/project/advers/train-nonlocal.sh
```
- **本地** 下载
```ps1
scp -r `
  ubuntu@192.168.101.11:/home/ubuntu/project/advers `-pll
  /Users/Henryzhu/project/deeplearning-cv
```
- **本地** 下载训练好的模型
```ps1
scp -r `
  ubuntu@192.168.101.11:/home/ubuntu/project/advers/checkpoints `
  /Users/Henryzhu/project/deeplearning-cv/advers/checkpoints
scp -r `
  ubuntu@192.168.101.11:/home/ubuntu/project/advers/logs `
  /Users/Henryzhu/project/deeplearning-cv/advers/logs
scp -r `
  ubuntu@192.168.101.11:/home/ubuntu/project/advers/runs `
  /Users/Henryzhu/project/deeplearning-cv/advers/runs
```

# tmux
## 新建与关闭
- 新建 session
```bash
tmux
tmux new -s <session-name> 
```
- 离开 session
```bash
tmux detach
```
**快捷键** `ctrl+b` `d`

- 进入 session
```bash
tmux attach -t <session-name>
```

- 查看 session 列表
```bash
tmux ls
```

关闭 session
- 关闭 session
```bash
tmux kill-session -t <session-name>
```
**快捷键** `ctrl+d`

## 窗口
- 切割窗格
```bash
# 上下切割窗格
tmux split-window
# 左右切割窗格
tmux split-window -h
```
**快捷键** 上下切割窗格 `ctrl+b` `%`  
**快捷键** 左右切割窗格 `ctrl+d` `"`

- 关闭当前的窗格

快捷键 关闭窗格通常使用 `ctrl+b` `x`

- 窗格显示时间

快捷键 `ctrl+b` `t` 将会把在当前的窗格当中显示时钟，非常酷炫的一个功能，点击 enter (回车键将会复原)。




# Tensorboard
```bash
tensorboard --logdir runs --host 192.168.101.11 --port 6606
```

# CAM
[torch-cam](https://github.com/frgfm/torch-cam)


# 运行
运行全部攻击
```bash
bash train.sh
```

# 代码细节
### 基础代码
- [基本的训练代码](train.py)，基于resnet

### FGSM
[FGSM](attack-fgsm.py)
```bash
python3 attack-fgsm.py
```
- `backbone` ResNet
- `parameter`
  - `epsilon`	沿着梯度的步长系数

FGSM(Fast Gradient Sign Method)，快速梯度符号法，是一种白盒攻击。
FGSM的攻击表达式如下：
$$
x^{'}=x+\varepsilon \cdot sign(\nabla_x J(x, y))
$$
其中$x$是输入数据，$y$输出数据，$sign$是符号函数，$\varepsilon$是攻击步长
FGSM通过对

FGSM(Fast Gradient Sign Method)，快速梯度下降法

在白盒环境下，通过求出模型对输入的导数，然后用符号函数得到其具体的梯度方向，接着乘以一个步长，得到的“扰动”加在原来的输入上就得到了在FGSM攻击下的样本。
$$
x^{'}=x+\varepsilon \cdot sign(\triangledown_x J(x, y))
$$
攻击成功就是模型分类错误，就模型而言，就是加了扰动的样本使得模型的$loss$增大。而所有基于梯度的攻击方法都是基于让$loss$增大这一点来做的。可以仔细回忆一下，在神经网络的反向传播当中，我们在训练过程时就是沿着梯度方向来更新更新$w$，$b$的值。这样做可以使得网络往$loss$减小的方向收敛。

$$
\begin{aligned}
    W_{ij}^{(l)}    &= W_{ij}^{(l)}-\alpha \frac{\partial}{\partial  W_{ij}^{(l)}}J(W,b) \\
    b_{i}^{(l)}     &= b_{i}^{(l)}-\alpha \frac{\partial}{\partial  b_{i}^{(l)}}J(W,b)
\end{aligned}
$$

那么现在我们既然是要使得$loss$增大，而模型的网络系数又固定不变，唯一可以改变的就是输入，因此我们就利用$loss$对输入求导从而“更新”这个输入。

### FGM
[FGM](attack-fgm.py)，基于resnet
```bash
python3 attack-fgm.py
```
### PGD
[PGD](attack-pgd.py)
```bash
python3 attack-pgd.py
```
- `backbone` ResNet
- `parameter`
  - `epsilon`	    原始样本的灰度偏移比例
  - `epsilon_iter`	梯度步长的改变比例
  - `num_steps`	    迭代次数

PGD全称是Projected Gradient descent。目的是为解决FGSM和FGM中的线性假设问题，使用PGD方法来求解内部的最大值问题。 PGD是一种迭代攻击，相比于普通的FGSM和FGM 仅做一次迭代，PGD是做多次迭代，每次走一小步，每次迭代都会将扰动投射到规定范围内。
$$
g_t=\triangledown X_t(L(f_\theta (X_t), y))
$$
$g_t$ 表示t时刻的损失关于$t$时刻输入的梯度。
$$
X_{t+1} = \prod _{X+S}(X_t+\varepsilon (\frac{g_t}{|| g_t ||}))
$$
$t+1$时刻输入根据t时刻的输入及t时刻的梯度求出。$\prod_{(X+S)}$的意思是，如果扰动超过一定的范围，就要映射回规定的范围S内。

由于每次只走很小的一步，所以局部线性假设基本成立。经过多步之后就可以达到最优解，也就是达到最强的攻击效果。同时使用PGD算法得到的攻击样本，是一阶对抗样本中最强的。这里所说的一阶对抗样本是指依据一阶梯度的对抗样本。如果模型对PGD产生的样本鲁棒，那基本上就对所有的一阶对抗样本都鲁棒。实验也证明，利用PGD算法进行对抗训练的模型确实具有很好的鲁棒性。

PGD虽然简单，也很有效，但是存在一个问题是计算效率不高。不采用提对抗训练的方法m次迭代只会有$m$次梯度的计算，但是对于PGD而言，每做一次梯度下降（获取模型参数的梯度，训练模型），都要对应有$K$步的梯度提升（获取输出的梯度，寻找扰动）。所以相比不采用对抗训练的方法，PGD需要做$m(K+1)$次梯度计算。














```txt
                                step             acc
1/255=0.003922	4/255=0.015686	40	0.924000	0.645500	0.023226	0.137002
1/255=0.003922	4/255=0.015686	20	0.924000	0.645500	0.023226	0.137227
1/255=0.003922	4/255=0.015686	10	0.924000	0.646000	0.023226	0.137298
1/255=0.003922	4/255=0.015686	5	0.924000	0.621500	0.023226	0.151619

1/255=0.003922	16/255=0.062745	40	0.924000	0.645500	0.023226	0.137126
1/255=0.003922	16/255=0.062745	20	0.924000	0.645500	0.023226	0.137220
1/255=0.003922	16/255=0.062745	10	0.924000	0.646000	0.023226	0.137231
1/255=0.003922	16/255=0.062745	5	0.924000	0.621500	0.023226	0.151622

1/255=0.003922	64/255=0.250980	40	0.924000	0.645000	0.023226	0.137114
1/255=0.003922	64/255=0.250980	20	0.924000	0.647000	0.023226	0.137250
1/255=0.003922	64/255=0.250980	10	0.924000	0.646000	0.023226	0.137271
1/255=0.003922	64/255=0.250980	5	0.924000	0.621500	0.023226	0.151622

4/255=0.015686	1/255=0.003922	40	0.924000	0.001500	0.023226	2.069437
4/255=0.015686	1/255=0.003922	20	0.924000	0.002500	0.023226	1.990766
4/255=0.015686	1/255=0.003922	10	0.924000	0.005500	0.023226	1.816244
4/255=0.015686	1/255=0.003922	5	0.924000	0.028500	0.023226	1.385023
```