'''
	Generate a .yaml template for network pruning.
	Need to import the network first.
'''

import os
from network import lenet, resnet, custom_network
import yaml

model = custom_network.ConvNet(in_channels=3)  # CIFAR-10


def prune_config(model):
    model_name = model.__class__.__name__
    yaml_name = os.path.join('profile', model_name + '.yaml')
    with open(yaml_name, 'w') as f:
        f.write('prune_ratios:\n')
    for key, value in model.state_dict().items():
        if len(value.shape) >= 2:
            with open(yaml_name, 'a') as f:
                f.write('#   ' + key + ':  #' + str(list(value.shape)) + '\n#     0.\n')


prune_config(model)