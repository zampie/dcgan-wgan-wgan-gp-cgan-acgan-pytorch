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

    image_size = 128  # All images will be resized to this size using a transformer.
    n_samples = 8

    samples_path = './DCGAN_Samples_1'
    ckpt_path = './DCGAN_Samples_1/checkpoint_iteration_2000.tar'
    os.makedirs(samples_path, exist_ok=True)

    latent_dim = 100  # Size of z latent vector (i.e. size of generator input)
    nc = 3  # Number of channels in the training images. For color images this is 3
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator

    ngpu = 0  # Number of GPUs available. Use 0 for CPU mode.
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Set random seem for reproducibility
    # manualSeed = random.randint(1, 10000) # use if you want new results
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    # manualSeed = 999
    # random.seed(manualSeed)
    sample_noise = torch.randn(n_samples, latent_dim, device=device)

    # np.random.seed(manualSeed)
    # sample_noise1 = np.random.rand(64,latent_dim)
    # np.random.seed(manualSeed)
    # sample_noise2 = np.random.rand(64,latent_dim)
    # sample_noise1 == sample_noise2
    #
    # torch.manual_seed(manualSeed)
    # sample_noise1 = torch.randn(64, latent_dim, device=device)
    # torch.manual_seed(manualSeed)
    # sample_noise2 = torch.randn(64, latent_dim, device=device)
    # sample_noise1 == sample_noise2

    class Reshape(nn.Module):
        def __init__(self, *args):
            super(Reshape, self).__init__()
            self.shape = args

        def forward(self, x):
            return x.view((x.size(0),) + self.shape)

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

    print(net_G)
    print("Loading checkpoint...")
    checkpoint = torch.load(ckpt_path)
    # sample_noise = checkpoint['sample_noise'].cpu()
    net_G.load_state_dict(checkpoint['netG_state_dict'])
    net_G.eval()

    samples = net_G(sample_noise)

    vutils.save_image(samples, os.path.join(samples_path, 'sample.jpg'), padding=2, normalize=True)
