# Filter Pruning

- Model: `custom_network.py`
- Pretrained model: `cifar10_custom_83.12.pt`
- Pruned model with zeros: `state-dict_cifar10_custom_filter_top1-83.880.pt`
- Pruned model without zeros: `cifar10_custom_filter_top1-83.880_pruned.pt`
- Pruning configuration: `ConvNet.yaml`
- Indices of retained weights in pruned model: `cifar10_custom_filter_top1-83.880_ind.npz`
- Save & Load indices: `store_ind.py`, `load_ind.py`

Load network architecture:
```python
from custom_network import ConvNet
model = ConvNet(in_channels=3)  # for CIFAR-10
```

# Save and load indices of preserved weights

Run `store_ind.py` with the pruned model with zeros, which will generate the pruned model (`.pt`) without zeros and indices of retained filters (`.npz`) for each convolutioanl weight matrix.

```sh
> python store_ind.py
conv2.weight density (non-zero ratio): 26/64=40.6250%
Retained indices: [ 2  3  4  6  7 13 15 24 26 31 34 35 38 41 42 43 44 47 51 53 55 56 57 60
 62 63]

conv3.weight density (non-zero ratio): 51/128=39.8438%
Retained indices: [  0   4   5   6   9  12  13  16  20  22  24  27  28  31  32  35  36  40
  43  46  47  50  52  54  59  61  62  66  67  68  74  77  81  82  83  85
  86  87  88  89  91  93  94  96 100 101 109 112 121 125 127]
```

Run `load_ind.py` to view the retained indices in `.npz` file:
```sh
> python load_ind.py
conv2.weight, 26 filters preserved
[ 2  3  4  6  7 13 15 24 26 31 34 35 38 41 42 43 44 47 51 53 55 56 57 60
 62 63]
conv3.weight, 51 filters preserved
[  0   4   5   6   9  12  13  16  20  22  24  27  28  31  32  35  36  40
  43  46  47  50  52  54  59  61  62  66  67  68  74  77  81  82  83  85
  86  87  88  89  91  93  94  96 100 101 109 112 121 125 127]
```
