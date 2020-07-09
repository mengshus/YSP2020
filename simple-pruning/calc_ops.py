'''
	Calculate the number of parameters and operations before and after pruning.
	Show the total number of a model and also the number of each convolutional or fully-connected layer.
	Need to specify the model, the number of channels of input, and the name of .yaml file in the `profile` folder.
	Can be run before pruning by just setting up a .yaml file to estimate the numbers 
		to help obtain a desired overall pruning ratio for the model.
'''
import os
import torch
from network import lenet_mnist, resnet_cifar
from custom_profile import profile_prune

model = lenet_mnist.LeNet5()
num_channels = 1
# dummy_input is an input variable generated randomlly only for calculating the number of operations.
# The first 1 means the batch size is 1. num_channels is 1 for mnist, and 3 for cifar. (32, 32) is the image size.
dummy_input = torch.randn(1, num_channels, 32, 32)
yaml_file = 'lenet'

flops, params, flops_prune, params_prune = profile_prune(model, inputs=(dummy_input, ), \
    prune=True, mode=1, file=os.path.join('profile', yaml_file + '.yaml'))