import argparse

# Training settings
parser = argparse.ArgumentParser(description='PyTorch pruning for LeNet5 and ResNet18 on MNIST or CIFAR-10')
parser.add_argument('--load-path', default='',
                    help='pretrained model path')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='network architecture, support lenet5 and resnet18',
                    choices=['lenet', 'resnet18'])
parser.add_argument('--dataset', default='cifar10',
                    help='support MNIST and CIFAR-10',
                    choices=['mnist', 'cifar10'])
parser.add_argument('--sparsity-type', type=str, default='filter',
                    help='pruning scheme, support filter/column/channel pruning',
                    choices=['filter', 'channel', 'column'])
parser.add_argument('--config-file', type=str, default='lenet',
                    help ='config file name, file at profile/<filename>.yaml')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1024)')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--epoch-prune', type=int, default=10, metavar='N',
                    help='number of interval epochs to update the zero weights (default: 10)')
parser.add_argument('--optmzr', type=str, default='adam', 
                    help='optimizer (default: adam), support SGD and Adam',
                    choices=['sgd', 'adam'])
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='initial learning rate (default: 1e-2)')
parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='lr scheduler, support step and cosine schedulers',
                    choices=['default', 'cosine'])
parser.add_argument('--lr-step', type=int, default=30, metavar='N',
                    help='number of interval epochs to decrease learning rate (default: 30)')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='value to be multiplied by the learning rate,' + 
                    ' each time after step_size epoch(s), new_lr = lr * gamma')

parser.add_argument('--resume',  action='store_true', default=False,
                    help='resume from last epoch if model exists')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='number of interval batches to update log of training status')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA training')