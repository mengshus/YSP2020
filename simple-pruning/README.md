`python pruning.py -h` will show the help text.

The `bash scripts/pruning.sh` provides the following sample command to run the pruning code.
```sh
CUDA_VISIBLE_DEVICES=1 python pruning.py --arch lenet --dataset mnist --load-path checkpoint/pretrained/mnist_lenet_top1-98.700.pt --sparsity-type filter --config-file lenet --optmzr sgd --epochs 100 --epoch-prune 5 --batch-size 256 --test-batch-size 1024 --lr 1e-3 --lr-decay 20 --lr-scheduler cosine
```
This command runs using the `lenet` architecture and `mnist` dataset. The loading path specified by `--load-path` is the path where the pretrained model is located, like `<some_root_path>/lenet_training/checkpoint/mnist_lenet_best.pt`. I'm used to put all the pretrained models into a folder called `pretrained` in the `checkpoint` folder.

Import the network before using. A WHOLE EXAMPLE IS PROVIDED AT LAST.

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

Run the `misc.py` file to get a `.yaml` template that will be stored into `profile` folder. The file name is the network class name (e.g., `LeNet5`). The network structure also needs to be imported and the model needs to be specified, like in `pruning.py`.
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

Usually we do not prune the first layer and the last layer.

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
