import os
import sys
import time
from time import strftime
import shutil
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import numpy as np
from numpy import linalg as LA
import yaml
from network import lenet, resnet, custom_network
from utils import *

from custom_profile import profile_prune
import copy

import argparse
from options import parser
args = parser.parse_args()

ckpt_name = '{}_{}_{}'.format(args.dataset, args.arch, args.sparsity_type)
args.ckpt_dir = os.path.join('checkpoint', ckpt_name)
if not args.resume and os.path.exists(args.ckpt_dir):
    i = 1
    while os.path.exists(args.ckpt_dir + '_v{}'.format(i)):
        i += 1
    os.rename(args.ckpt_dir, args.ckpt_dir + '_v{}'.format(i))
os.makedirs(args.ckpt_dir, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(os.path.join(args.ckpt_dir, '{}_{}_{}.log'.format( \
    ckpt_name, 'pruning', strftime('%m%d%Y-%H%M'))), 'w'))
global print
print = logger.info

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.backends.cudnn.benchmark = True  # will result in non-determinism

args.dataset = args.dataset.lower()

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

# set up model architecture
args.arch = args.arch.lower()
if args.arch == 'lenet':
    model = lenet.LeNet5(in_channels)
elif args.arch == 'resnet18':
    model = resnet.ResNet18(in_channels)
elif args.arch == 'custom':
    model = custom_network.ConvNet(in_channels)

model_cpu = copy.deepcopy(model)
if use_cuda:
    model.cuda()
    

def load_model(model, checkpoint, optimizer, first=False):
    # baseline model for pruning, pruned model for retrain
    try:
        state_dict = checkpoint['state_dict']
    except:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def main():
    print(' '.join(sys.argv))

    print('General config:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()

    optimizer = None
    if args.optmzr == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optmzr == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    scheduler = None
    if args.lr_scheduler == 'default':
        # each time after step_size epoch(s), new_lr = lr * gamma
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)
    else:
        raise Exception('unknown lr scheduler')

    # load pretrained model to be pruned
    if not args.resume:
        load_path = args.load_path
        print('>_ Loading model from {}\n'.format(load_path))
    else:
        load_path = os.path.join(args.ckpt_dir, '{}.pt'.format(ckpt_name))
    
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
    else:
        exit('Checkpoint does not exist.')

    if use_cuda:
        model.cuda()
        load_model(model, checkpoint, optimizer, first=True)

    start_epoch = 1
    loss_list = []
    acc_list = [0.]
    best_epoch = 0
    if args.resume:
        start_epoch = checkpoint['epoch'] + 1
        try:
            checkpoint = torch.load(load_path.replace('.pt', '_best.pt'), map_location='cpu')
            best_epoch = checkpoint['epoch']
            best_top1 = checkpoint['top1']
        except:
            pass

    # restore scheduler
    for epoch in range(1, start_epoch):
        for _ in range(len(train_loader)):
            scheduler.step()

    # load .yaml configuration file with pruning ratios for layers in the model
    with open(os.path.join('profile', args.config_file + '.yaml'), 'r') as stream:
        raw_dict = yaml.full_load(stream)
        prune_cfg = raw_dict['prune_ratios']
    print('Prune config:')
    for k, v in prune_cfg.items():
        print('\t{}: {}'.format(k, v))
    print('')

    save_path = os.path.join(args.ckpt_dir, '{}.pt'.format(ckpt_name))

    for epoch in range(start_epoch, args.epochs + 1):
        print('')
        idx_loss_dict = train(train_loader, criterion, optimizer, scheduler, epoch, args)

        do_prune = (epoch - 1) % args.epoch_prune == 0 or epoch == args.epochs
        if do_prune:
            print('Before pruning:')
        _, top1 = test(model, criterion, test_loader)

        if do_prune:
            # perform pruning
            for (name, W) in model.named_parameters():
                if name not in prune_cfg:  # ignore layers that are not pruned
                    continue
                cuda_pruned_weights = None
                _, cuda_pruned_weights = mask(args, W, prune_cfg[name])  # get sparse model in cuda
                W.data = cuda_pruned_weights  # replace the data field in variable

            print('After pruning:')
            loss, top1 = test(model, criterion, test_loader)

            best_top1 = max(acc_list)
            is_best = top1 > best_top1

            save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'top1': top1,
            },
            is_best, save_path)
            if is_best:
                best_top1 = top1
                best_epoch = epoch

            print('\nBest Acc@1 {:.3f}%   Best epoch {}\n'.format(best_top1, best_epoch))

            loss_list.append(loss)
            acc_list.append(top1)

            # print the current sparsity
            in_channels = 1 if args.dataset == 'mnist' else 3
            dummy_input = torch.randn(1, in_channels, 32, 32)
            print('Current sparsity:')
            flops, params, flops_prune, params_prune = profile_prune(model_cpu, inputs=(dummy_input, ), \
                prune=True, mode=0, file=save_path, print=print)
            print('')

    os.rename(save_path.replace('.pt', '_best.pt'), \
        save_path.replace('.pt', '_epoch-{}_top1-{:.3f}.pt'.format(best_epoch, best_top1)))

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    epoch_list = list(range(1, args.epochs+1, args.epoch_prune)) + [args.epochs]

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
    ax2.plot(epoch_list, acc_list[1:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(args.epoch_prune))
    plt.savefig(os.path.join(ckpt_dir, ckpt_name +'_results.pdf'))


def train(train_loader, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    idx_loss_dict = {}

    # switch to train mode
    model.train()
    
    end = time.time()
    epoch_start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        scheduler.step()

        if use_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)

        ce_loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, _ = accuracy(output, target, topk=(1,5))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        ce_loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print('({0}) lr [{1:.6f}]   '
                  'Epoch [{2}][{3:3d}/{4}]   '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                  'Acc@1 {top1.val:7.3f}% ({top1.avg:7.3f}%)'
                  .format(args.optmzr, current_lr,
                   epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1))
        if i % 100 == 0:
            idx_loss_dict[i] = losses.avg
    print('[Train] Loss {:.4f}   Acc@1 {:.3f}%   Time {}'.format(
        losses.avg, top1.avg, int(time.time() - epoch_start_time)))
    return idx_loss_dict


def test(model, criterion, test_loader):
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = len(test_loader.dataset)
    epoch_start_time = time.time()
    with torch.no_grad():
        for input, target in test_loader:
            if use_cuda:
                input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    top1 = 100. * float(correct) / float(total)
    print('Test Loss {:.4f}   Acc@1 {}/{} ({:.3f}%)   Time {}' \
        .format(losses.avg, correct, total, top1, int(time.time() - epoch_start_time)))
    return losses.avg, top1


def mask(args, weight_in, prune_ratio):
    '''
    weight pruning [filter, channel, column]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose filter/channel/column that have lowest l2 norms (equivalent to absolute weight here) are set to zero

    '''
    weight = weight_in.cpu().detach().numpy()  # convert cpu tensor to numpy
    percent = prune_ratio * 100

    if args.sparsity_type == 'filter':
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm <= percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        weight = weight.reshape(shape)
        above_threshold = above_threshold.astype(np.float32)
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()

    elif args.sparsity_type == 'channel':
        shape = weight.shape
        weight3d = weight.reshape(shape[0], shape[1], -1)
        channel_l2_norm = LA.norm(weight3d, 2, axis=(0,2))
        percentile = np.percentile(channel_l2_norm, percent)
        under_threshold = channel_l2_norm <= percentile
        above_threshold = channel_l2_norm > percentile
        weight3d[:,under_threshold,:] = 0
        above_threshold = above_threshold.astype(np.float32)
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    
    elif args.sparsity_type == 'column':
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm <= percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


if __name__ == '__main__':
    main()
