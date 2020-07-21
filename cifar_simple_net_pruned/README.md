# Filter Pruning

- Model: `custom_network.py`
- Pretrained model: `cifar10_custom_83.12.pt`
- Pruned model with zeros: `state-dict_cifar10_custom_filter_top1-83.880.pt`
- Pruned model without zeros: `cifar10_custom_filter_top1-83.880_pruned.pt`
- Pruning configuration: `ConvNet.yaml`
- Indices of retained weights in pruned model: `cifar10_custom_filter_top1-83.880_ind.npz`
- How to load indices: `load_ind.py`

Load network architecture:
```python
from custom_network import ConvNet
model = ConvNet(in_channels=3)  # for CIFAR-10
```
