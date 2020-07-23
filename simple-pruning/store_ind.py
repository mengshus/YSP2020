## Extract indices of unpruned filters of the model in filter pruning.
import torch
import numpy as np

model_path = 'checkpoint/state-dict_cifar10_custom_filter/cifar10_custom_filter_epoch-100_top1-75.490.pt'
checkpoint = torch.load(model_path, map_location='cpu')
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

index_dict = {}
pruned_state_dict = {}
for k, v in checkpoint.items():
    if len(v.shape) <= 2:  # not conv
        pruned_state_dict[k] = v
        continue
    v_2d = v.reshape(v.shape[0], -1)
    retain_ind = (torch.sum(torch.abs(v_2d), dim=1) != 0).nonzero() \
        .squeeze().numpy()
    pruned_v = v[retain_ind]
    if len(pruned_v) != len(v):  # unpruned layer
        index_dict[k] = retain_ind
        print('{} density (non-zero ratio): {:d}/{:d}={:.4f}%'.format(
            k, len(retain_ind), len(v), float(len(retain_ind))/len(v)*100))
        print('Retained indices: {}'.format(retain_ind))
        print('')
    pruned_state_dict[k] = pruned_v

print('Save retained indices for pruned weights')
np.savez(model_path.replace('.pt', '_ind.npz'), **index_dict)
print('Save pruned model without zeros')
torch.save(pruned_state_dict, model_path.replace('.pt', '_pruned.pt'))

# ## Load saved indices
# load_ind = np.load(model_path.replace('.pt', '_ind.npz'))
# for k in load_ind.files:
#     print(k)
#     print(load_ind[k])