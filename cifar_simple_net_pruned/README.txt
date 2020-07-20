---------- Filter Pruning ----------

- Model: `custom_network.py`
- Pretrained model: `cifar10_custom_83.12.pt`
- Pruned model with zeros: `state-dict_cifar10_custom_filter_epoch-100_top1-75.490.pt`
- Pruned model without zeros: `cifar10_custom_filter_epoch-100_top1-75.490_pruned.pt`
- Pruning configuration: `ConvNet.yaml`
- Indices of retained weights in pruned model: `cifar10_custom_filter_epoch-100_top1-75.490_ind.npz`
- How to load indices: `load_ind.py`

Use `cifar10_custom_filter_epoch-100_top1-75.490_pruned.pt` and `cifar10_custom_filter_epoch-100_top1-75.490_ind.npz` in hardware implementation.