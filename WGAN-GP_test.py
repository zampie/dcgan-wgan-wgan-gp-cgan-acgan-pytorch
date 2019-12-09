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
from torch.autograd import Variable
import torch.autograd as autograd

import numpy as np
import matplotlib.pyplot as plt
import torch.functional as F

if __name__ == '__main__':

    batch_size = 64  # Batch size during training
    image_size = 128  # All images will be resized to this size using a transformer.

    samples_path = './WGAN-GP_Samples'
    os.makedirs(samples_path, exist_ok=True)

    is_load = False
    ckpt_path = './WGAN-GP_Samples/checkpoint_iteration_54000.tar'

    latent_dim = 100  # Size of z latent vector (i.e. size of generator input)
    nc = 3  # Number of channels in the training images. For color images this is 3
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers

    ngpu = 0  # Number of GPUs available. Use 0 for CPU mode.
    workers = 0  # Number of workers for dataloader
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    # Pytorch 没有nn.Reshape, 且不推荐使用 Why？？
    class Reshape(nn.Module):
        def __init__(self, *args):
            super(Reshape, self).__init__()
            self.shape = args

        def forward(self, x):
            # 自动取得batch维
            return x.view((x.size(0),) + self.shape)
            # 若使用下式，batch的维数只能用-1代指
            # return x.view(self.shape)


    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(

                nn.Linear(latent_dim, ngf * 8 * (image_size // 16) ** 2),
                Reshape(ngf * 8, image_size // 16, image_size // 16),
                nn.BatchNorm2d(ngf * 8),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x (image_size//8) x (image_size//8)

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x (image_size//4) x (image_size//4)

                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x (image_size//2) x (image_size//2)

                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x image_size x image_size
            )

        def forward(self, input):
            return self.main(input)


    net_G = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net_G = nn.DataParallel(net_G, list(range(ngpu)))
        # net_D = nn.DataParallel(net_D, list(range(ngpu)))

    print(net_G)
    # print(net_D)

    # Training Loop
    # Lists to keep track of progress
    print("Loading checkpoint...")
    checkpoint = torch.load(ckpt_path)

    net_G.load_state_dict(checkpoint['netG_state_dict'])

    net_G.eval()

    # manualSeed = 1200
    # torch.manual_seed(manualSeed)
    # # manual_seed的作用期很短
    # sample_noise = torch.randn(64, latent_dim, device=device)

    # sample_noise = checkpoint['sample_noise'].to(device)

    # ----------------------------------------------------------------------------------------
    # 噪声插值
    n_sample = 10
    sample_noise = torch.randn(n_sample, latent_dim, device=device)
    for i in range(n_sample):
        sample_noise[i] = torch.lerp(sample_noise[0], sample_noise[-1], i / (n_sample - 1))

    samples = net_G(sample_noise)
    vutils.save_image(samples, os.path.join(samples_path, 'sample.jpg'), padding=2, nrow=n_sample, normalize=True)

    # ----------------------------------------------------------------------------------------
    # 平均噪声
    n_sample = 8
    z1 = torch.randn(n_sample, latent_dim, device=device)
    z2 = torch.randn(n_sample, latent_dim, device=device)
    z3 = (z1 + z2) * 0.5

    sample_noise = torch.cat((z1, z2, z3))
    samples = net_G(sample_noise)
    vutils.save_image(samples, os.path.join(samples_path, 'plus.jpg'), padding=2, nrow=n_sample, normalize=True)

    # ----------------------------------------------------------------------------------------
    # 矢量运算
    n_sample = 8
    z1 = torch.randn(n_sample, latent_dim, device=device)
    z2 = torch.randn(n_sample, latent_dim, device=device)
    z3 = torch.randn(n_sample, latent_dim, device=device)

    z4 = z1 + z2 - z3

    sample_noise = torch.cat((z1, z2, z3, z4))
    samples = net_G(sample_noise)
    vutils.save_image(samples, os.path.join(samples_path, 'plus_minus.jpg'), padding=2, nrow=n_sample, normalize=True)