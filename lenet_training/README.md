Setup a new Anaconda environmnet by `conda create --name py36 python=3.6`.
Activate the environment by **`conda activate py36`** and deactivate it by **`conda deactivate`** after use.

The required packages `torch` and `torchvision` can be installed by `conda install --file requirements.txt`.
To install additional packages, use `conda install <package_name>`.

A sample running command for running python files is shown in `train.sh`. This can be run by **`bash train.sh`**.

We have 4 GPUs on the server with IDs 0, 1, 2, and 3. By adding `CUDA_VISIBLE_DEVICES=<gpu_id>`, we can designate a GPU to run the code. **Please use GPU with ID 1 or 3**, like `CUDA_VISIBLE_DEVICES=1 python train.py`.

---

Useful links:
- PyTorch neural network tutorial https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- Online book explaning neural networks http://neuralnetworksanddeeplearning.com/
- Standford CS231n class https://cs231n.github.io/

---

06/30/2020:

Add plot commands and training accuracy of batches in the training file as `train_v2.py`.

Install the `matplotlib` package by `conda install matplotlib` to plot the loss, accuracy, etc.
Matplotlib tutorial https://matplotlib.org/tutorials/introductory/pyplot.html

See https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html to configure the convolutional layers.
