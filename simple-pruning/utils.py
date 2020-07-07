import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import shutil
    

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pt', '_best.pt'))

        
class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    '''Computes the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs'''
    # only in masked retrain

    # adjust learning rate
    if args.warmup and epoch - 1 <= args.warmup_epochs:
        lr = args.warmup_lr + (args.lr - args.warmup_lr) / args.warmup_epochs * (epoch - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    args.lr_decay = max(1, int(args.epochs * 0.2))
    #lr = args.lr * (0.3 ** (epoch // args.lr_decay))
    lr = args.lr * (0.5 ** ((epoch - 1) // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class GradualWarmupScheduler(_LRScheduler):
    ''' Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    '''

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)