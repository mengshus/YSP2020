'''lenet tutorial from
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):

    def __init__(self, in_channels=1):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # conv1-relu-maxpool-conv2-relu-maxpool-fc1-relu-fc2-relu-fc3
        # 32x32 input
        x = self.conv1(x)  # 30x30 feature
        x = F.relu(x)
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x, (2, 2))  # 15x15 feature
        x = self.conv2(x)  # 13x13 feature
        x = F.relu(x)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(x, 2)  # 6x6 feature
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def lenet(in_channels=1):
    return LeNet5(in_channels)


if __name__ == '__main__':
    net = lenet()
    print(net)