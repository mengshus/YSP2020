`python pruning.py -h` will show the help text.

The `bash scripts/pruning.sh` provides the following sample command to run the pruning code.
```sh
CUDA_VISIBLE_DEVICES=1 python pruning.py --arch lenet --dataset mnist --load-path checkpoint/pretrained/mnist_lenet_top1-98.700.pt --sparsity-type filter --config-file lenet --optmzr sgd --epochs 100 --epoch-prune 5 --batch-size 256 --test-batch-size 1024 --lr 1e-3 --lr-decay 20 --lr-scheduler cosine
```
This command runs using the `lenet` architecture and `mnist` dataset. The loading path specified by `--load-path` is the path where the pretrained model is located, like `<some_root_path>/lenet_training/checkpoint/mnist_lenet_best.pt`. I'm used to put all the pretrained models into a folder called `pretrained` in the `checkpoint` folder.

Import the network before using. A WHOLE EXAMPLE IS PROVIDED AT LAST.

# Update on 07/09/2020: observe the overall pruning ratio

Run `calc_ops.py` to estimate the number of parameters and operations before performing pruning to help obtain a desired overall pruning ratio for the model. This provides the numbers before and after pruning for comparison. In this file, you need to specify the model, the number of channels in input (1 for MNIST and 3 for CIFAR-10), and the name of `.yaml` file (like `lenet`) stored in `profile` folder.

The numbers will be shown in the following format:
```sh
layer name or total
parameters remaining/total (overall compression ratio): 45943/81194 (1.77x)
flops (operations) remaining/total (overall compression ratio): 125467/274656 (2.19x)
```
Here is a whole example:
```sh
conv1
    params rmn/tot: 60/60    flops rmn/tot: 97200/97200

conv2
    params rmn/tot: 189/880    flops rmn/tot: 62654/292032

fc1
    params rmn/tot: 34680/69240    flops rmn/tot: 69240/138240

fc2
    params rmn/tot: 10164/10164    flops rmn/tot: 20160/20160

fc3
    params rmn/tot: 850/850    flops rmn/tot: 1680/1680

total:
    params rmn/tot: 45943/81194 (1.77x)    flops rmn/tot: 125467/274656 (2.19x)
```
The numbers are shown for each convolutional or fully-connected layer, and finally for the whole model. We would like to increase the overall compression ratios of parameters and operations.

Code is also added in `pruning.py` to observe the current pruning ratio in training and needs no modification. Since we perform "hard" pruning each time, the pruning ratio for each layer does not change, but the places of pruned weight values may change.

# Import the Network Structure

- Put the network structure file `<filename>.py` into `network` folder.
- In `pruning.py`, import the network file using `from network import <filename>` (e.g., if the network file is `lenet.py`, then use `from network import lenet`).
- In model setup, specify the model according to the network and the dataset as follows. The network is defined in a class in `<filename>.py`, like the `LeNet5` class in `lenet.py`.
```python
if args.arch == 'some_network':
    if args.dataset == 'some_dataset':
        model = <filename>.<class>  # e.g., lenet.LeNet5()
```
In the running command, specify the network by `--arch` and the dataset by `--dataset`.

# Pruning Configuration

Configure the `.yaml` file in `profile` folder by setting the pruning ratio for each layer. Pruning ratio should be from 0 to 1, e.g., if pruning ratio is set to 0.3, then 30% of weights of this layer will be set to zero in pruning.

In the running command, specify the `.yaml` file by `--config-file`. For a file `lenet.yaml` stored in `profile` folder, just use `--config-file lenet`.

Run the `gen_yaml.py` file to get a `.yaml` template that will be stored into `profile` folder automatically. The file name will be the network class name (e.g., `LeNet5`). The network structure also needs to be imported and the model needs to be specified, like in `pruning.py`.
The template looks like this:
```yaml
prune_ratios:
#   conv1.weight:  #[6, 1, 3, 3]
#     0.
#   conv2.weight:  #[16, 6, 3, 3]
#     0.
#   fc1.weight:  #[120, 576]
#     0.
#   fc2.weight:  #[84, 120]
#     0.
#   fc3.weight:  #[10, 84]
#     0.
```
It provides the name of the weight parameters in each layer and the pruning ratio is 0 be default. To set the pruning ratio of a certain layer, uncomment the line with the layer weight name and the next line with the number 0 and change the number. If we would like to prune 30% of the weight parameters in layer conv2, the file should be like this:
```yaml
prune_ratios:
#   conv1.weight:  #[6, 1, 3, 3]
#     0.
   conv2.weight:  #[16, 6, 3, 3]
     0.3
#   fc1.weight:  #[120, 576]
#     0.
#   fc2.weight:  #[84, 120]
#     0.
#   fc3.weight:  #[10, 84]
#     0.
```
Just leave the comment for the layer size as it is. It is printed for reference. Layers with larger layer size (number of output channels for filter pruning) can usually have larger pruning ratios.

All the convolutional layers and fully-connected layers are provided in the `.yaml` file to make it convenient to configure the pruning ratios.

Usually we do not prune the first layer and the last layer. For a fully-connected layer, filter pruning would remove the rows of the 2-D weight matrix.

# Save the Pruned Model

The pruned models and logs will be stored in `checkpoint` folder automatically. The logs include loss, accuracy and time consumed and will also be printed during training. Each time you run the code, a new model folder will be generated, and the old ones will be renamed to `<checkpoint_name>_vx`, where `x` is a number.

There will be two checkpoints, one is from the last training epoch, called `<dataset>_<network>_<sparsity_type>.pt` (e.g., `mnist_lenet_filter.pt`), and the other is the best model during training, called `<dataset>_<network>_<sparsity_type>_epoch-N_top1-xxx.pt` (e.g., `mnist_lenet_filter_epoch-81_top1-98.550.pt`), where `N` is the epoch from which the best model is obtained, and `xxx` is the best accuracy during training.

# Arguments that can be modified

- path of pretrained model to be loaded: `--load-path`
- model architecture: `--arch`, choose from ['lenet', 'resnet18']
- dataset: `--dataset`, choose from ['mnist', 'cifar10']
- pruning scheme: `--sparsity-type`, choose from ['filter', 'column', 'channel']
- pruning configuration file: `config-file`, prestored at `profile/<filename>.yaml`
- total number of epochs: `--epochs`
- number of interval epochs to prune the model: `--epoch-prune`
- optimizer: `--optmzr`, choose from ['sgd', 'adam']
- initial learning rate: `--lr`
- learning rate scheduler: `--lr-scheduler`, choose from ['default', 'cosine']
- number of interval epochs to decrease the learning rate (only used in default scheduler): `--lr-step`
- decay value of learning rate: `--lr-decay`
- resume from the last saved model: `--resume`

# A Whole Example

- In `pruning.py`:
```python
from network import lenet  # for cifar10
if args.arch == 'lenet':
    if args.dataset == 'cifar10':
        model = lenet.LeNet5()
```
- In running command:
```sh
CUDA_VISIBLE_DEVICES=1 python pruning.py --arch lenet --dataset cifar10 --load-path checkpoint/pretrained/<pretrained_model_for_cifar10>.pt --sparsity-type filter --config-file lenet --optmzr sgd --epochs 100 --epoch-prune 5 --batch-size 256 --test-batch-size 1024 --lr 1e-3 --lr-decay 20 --lr-scheduler cosine
```
