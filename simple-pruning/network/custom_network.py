import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self, in_channels=1):
        super(ConvNet, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # 8x8 from image dimension
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # conv1-relu-conv2-relu-maxpool-conv3-relu-maxpool-fc1-relu-fc2-relu-fc3
        # 32x32 input
        x = self.conv1(x)  # 32x32 feature with kernel_size=3, padding=1
        x = F.relu(x)
        x = self.conv2(x)  # 32x32 feature
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 16x16 feature
        x = self.conv3(x)  # 16x16 feature
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 8x8 feature
        x = x.view(x.size(0), -1)  # flatten before FC
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def convnet(in_channels=1):
    return ConvNet(in_channels)


if __name__ == '__main__':
    net = convnet()
    print(net)