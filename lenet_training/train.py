'''train Lenet model on MNIST dataset
    modified from https://github.com/activatedgeek/LeNet-5/blob/master/run.py'''

from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', '-b', type=int, default=256)
parser.add_argument('--test-batch-size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-2,
                    help='learning rate')
parser.add_argument('--use-cuda', action='store_true', default=False,
                    help='use gpu device or not')
args = parser.parse_args()

data_train = MNIST('./data/mnist',
                    train=True,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=args.test_batch_size, num_workers=8)

# set model storing path
import os
ckpt_dir = 'checkpoint'
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_name = 'mnist_lenet'

model = LeNet5()
if args.use_cuda:
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    loss_list, batch_list = [], []
    for i, (input, target) in enumerate(data_train_loader):
        if args.use_cuda:
            input = input.cuda(non_blocking=True)
        if args.use_cuda:
            target = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        output = model(input)

        loss = criterion(output, target)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch {}, Batch: {}, Loss: {:f}'.format(epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()


def test():
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (input, target) in enumerate(data_test_loader):
        if args.use_cuda:
            input = input.cuda()
            target = target.cuda()
        output = model(input)
        avg_loss += criterion(output, target).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(target.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test) * 100
    print('Test Avg. Loss: {:f}, Accuracy: {:.3f}%\n'.format(avg_loss.detach().cpu().item(), acc))
    return acc


def main():
    best_acc = 0.
    for epoch in range(1, args.epochs+1):
        train(epoch)
        test_acc = test()

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
    print('Best Accuracy: {:.3f}%'.format(best_acc))


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename.replace('.pt', '_best.pt'))


if __name__ == '__main__':
    main()