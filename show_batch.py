import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torch.functional as F

if __name__ == '__main__':
    # data_path = "/dataset/Anime"
    data_path = "/datasets/celeba"

    image_size = 128
    batch_size = 64
    workers = 0

    dataset = dset.ImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers)

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0][:64], nrow=8, padding=0, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
