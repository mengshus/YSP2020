`python pruning.py -h` will show the help text.

Use `bash scripts/pruning.sh` to run the pruning code.

Configure the `.yaml` file in `profile` folder by setting the pruning ratios for each layer, pruning ratio should be from 0 to 1, e.g., pruning ratio is set to 0.3, then 30% of weights of this layer will be set to zero in pruning.

Usually we do not prune the first layer and the last layer.

The pruned models and logs will be stored in `checkpoint` folder.

Arguments that can be modified:
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
