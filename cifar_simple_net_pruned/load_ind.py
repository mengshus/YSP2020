import numpy as np
npz_file = 'cifar10_custom_filter_epoch-100_top1-75.490_ind.npz'
load_ind = np.load(npz_file)
for k in load_ind.files:
    print(k)
    print(load_ind[k])