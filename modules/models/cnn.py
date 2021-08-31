
from torch import nn
from torch import Tensor
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),nn.MaxPool2d(2))
        self.dense = nn.Sequential(nn.Linear(64 * 3 * 3, 128), nn.ReLU(),nn.Linear(128, 10))

    def forward(self, x:Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x