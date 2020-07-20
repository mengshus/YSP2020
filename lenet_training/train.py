'''train Lenet model on MNIST or CIFAR dataset
    modified from https://github.com/activatedgeek/LeNet-5/blob/master/run.py'''

import os
import sys

from lenet import LeNet5
from custom_network import ConvNet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import shutil
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='lenet',
                    help='network architecture, support lenet5 and custom network',
                    choices=['lenet', 'custom'])
parser.add_argument('--dataset', default='mnist',
                    help='support MNIST and CIFAR-10',
                    choices=['mnist', 'cifar10'])
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', '-b', type=int, default=256)
parser.add_argument('--test-batch-size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='use gpu device or not')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available
kwargs = {'num_workers': 12, 'worker_init_fn': np.random.seed(2020), 'pin_memory': True} if use_cuda else {}

## data loader
if args.dataset == 'mnist':
    data_train = datasets.MNIST('./data/mnist', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()]))
    data_test = datasets.MNIST('./data/mnist', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    in_channels = 1

elif args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar10', train=True, download=True,
                transform=transforms.Compose([
                               transforms.Pad(4),
                               transforms.RandomCrop(32),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar10', train=False, download=True,
                transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)

    in_channels = 3

# set model storing path
ckpt_dir = 'checkpoint'
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_name = '_'.join([args.dataset, args.arch])

if args.arch == 'lenet':
    model = LeNet5(in_channels=in_channels)
else:
    model = ConvNet(in_channels=in_channels)

if use_cuda:
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    loss_list, batch_list = [], []
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        output = model(input)

        loss = criterion(output, target)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        pred = output.detach().max(1)[1]
        total_correct = pred.eq(target.view_as(pred)).sum()
        acc = float(total_correct) / args.batch_size * 100

        if i % 10 == 0:
            print2log('Train - Epoch {}, Batch {}, Loss {:f}, Accuracy {:.3f}' \
                .format(epoch, i, loss.detach().cpu().item(), acc))

        loss.backward()
        optimizer.step()


def test():
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()
            output = model(input)
            avg_loss += criterion(output, target).float().sum()/input.size(0)
            pred = output.max(1)[1]
            total_correct += pred.eq(target.view_as(pred)).float().sum()/input.size(0)

    avg_loss /= len(test_loader)
    loss = avg_loss.detach().cpu().item()
    acc = float(total_correct) / len(test_loader) * 100
    print2log('Test Avg. Loss {:f}, Accuracy {:.3f}%\n'.format(loss, acc))
    return loss, acc


def main():
    best_acc = 0.

    loss_list = []
    acc_list = []

    for epoch in range(1, args.epochs+1):
        train(epoch)
        test_loss, test_acc = test()

        loss_list.append(test_loss)
        acc_list.append(test_acc)

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
        save_checkpoint(
        {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'accuracy': test_acc,
        },
        is_best, filename=os.path.join(ckpt_dir, '{}.pt'.format(ckpt_name)))
    print2log('Best Accuracy {:.3f}%'.format(best_acc))

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    epoch_list = range(1, args.epochs+1)

    ## plot loss and accuracy
    # https://matplotlib.org/gallery/api/two_scales.html
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('test loss', color=color)
    ax1.plot(epoch_list, loss_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('test acc', color=color)  # we already handled the x-label with ax1
    ax2.plot(epoch_list, acc_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.savefig(os.path.join(ckpt_dir, ckpt_name +'_results.pdf'))


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pt', '_best.pt'))


def print2log(content):
    print(str(content))
    with open(os.path.join(ckpt_dir, ckpt_name + '_log.txt'), 'a') as f:
        f.write(str(content) + '\n')


if __name__ == '__main__':
    with open(os.path.join(ckpt_dir, ckpt_name + '_log.txt'), 'w') as f:
        pass
    print2log('Command:')
    print2log(' '.join(sys.argv))
    print2log('')
    print2log('Training Arguments:')
    for k, v in sorted(vars(args).items()):
        print2log('   {}: {}'.format(k, v))
    print2log('')
    print2log('Network Architecture:')
    print2log(model)
    print2log('')
    main()