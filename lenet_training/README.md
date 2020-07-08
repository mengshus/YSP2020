Setup a new Anaconda environmnet by `conda create --name py36 python=3.6`.
Activate the environment by **`conda activate py36`** and deactivate it by **`conda deactivate`** after use.

The required packages `torch` and `torchvision` can be installed by `conda install --file requirements.txt`.
To install additional packages, use `conda install <package_name>`.

The following sample running command for running python files is shown in `train.sh`. This can be run by **`bash train.sh`**.
```
CUDA_VISIBLE_DEVICES=1 python train.py --epochs 20 --batch-size 256 --test-batch-size 1024 --lr 1e-2 --use-cuda
```

The model checkpoints will be stored in the `checkpoint` folder automatically. There are two checkpoint .pt file, one is from the last training epoch called `<dataset_name>_<network_name>.pt` (e.g., `mnist_lenet.pt`), and the other is the best model during the training process called `<dataset_name>_<network_name>_best.pt` (e.g., `mnist_lenet_best.pt`).

We have 4 GPUs on the server with IDs 0, 1, 2, and 3. By adding `CUDA_VISIBLE_DEVICES=<gpu_id>`, we can designate a GPU to run the code. **Please use GPU with ID 1 or 3**, like `CUDA_VISIBLE_DEVICES=1 python train.py`.

---

Useful links:
- PyTorch neural network tutorial https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- Online book explaning neural networks http://neuralnetworksanddeeplearning.com/
- Standford CS231n class https://cs231n.github.io/

---

06/30/2020:

Add plot commands and training accuracy of batches in the training file as `train_v2.py`.
**The plottings are stored in .pdf files in `checkpoint` folder.** Each time you run the code, the files in `checkpoint` will be overwritten. If you do not want this, you can modify the code to generate a new checkpoint folder in each run.

Install the `matplotlib` package by `pip install matplotlib` to plot the loss, accuracy, etc.
Matplotlib tutorial https://matplotlib.org/tutorials/introductory/pyplot.html

See https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html to configure the convolutional layers.

---

07/01/2020:

- MNIST dataset https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
- CIFAR-10 dataset https://pytorch.org/docs/stable/torchvision/datasets.html#cifar
- Example https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Concept of gradient descent and why use mini-batch https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size

Example of dataloaders for CIFAR-10:
```python
kwargs = {'num_workers': 12, 'worker_init_fn': np.random.seed(2020), 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
            transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
 
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, 
            transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
```
