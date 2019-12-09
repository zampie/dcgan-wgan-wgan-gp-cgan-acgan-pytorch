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
import torch.autograd as autograd
import time

if __name__ == '__main__':
    # -------------------------------------------------------------------------------------
    # 设置路径
    samples_path = './CGAN+_Samples_'
    ckpt_path = './CGAN+_Samples_/checkpoint_iteration_10000.tar'

    # -------------------------------------------------------------------------------------
    # 请保证参数与训练时一致
    image_size = 128  # All images will be resized to this size using a transformer.
    n_classes = 2

    latent_dim = 100  # Size of z latent vector (i.e. size of generator input)
    n_channels = 3  # Number of channels in the training images. For color images this is 3
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers

    # -------------------------------------------------------------------------------------
    # 一般使用CPU测试
    ngpu = 0  # Number of GPUs available. Use 0 for CPU mode.
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    # -------------------------------------------------------------------------------------
    # 定义网络
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
            self.latent_class_dim = 10  # 包含分类信息的噪声维数
            # 如果输入c为1维int型,升到2维到且第2维为latent_class_dim
            # self.emb = nn.Embedding(n_classes, self.latent_class_dim)
            # 如果输入c为one-hot，第2维扩张到latent_class_dim
            self.exp = nn.Linear(n_classes, self.latent_class_dim)
            self.main = nn.Sequential(

                nn.Linear(latent_dim + self.latent_class_dim, ngf * 8 * (image_size // 16) ** 2),
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

                nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x image_size x image_size
            )

        def forward(self, z, c):
            # c为一维int型
            # cat = torch.cat((z, self.emb(c)), 1)
            # c为one-hot
            cat = torch.cat((z, self.exp(c)), 1)
            return self.main(cat)


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.emb = nn.Embedding(n_classes, image_size * image_size)
            self.main = nn.Sequential(
                # input is (nc) x image_size x image_size
                nn.Conv2d(n_channels + 1, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x (image_size//2) x (image_size//2)

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x (image_size//4) x (image_size//4)

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x (image_size//8) x (image_size//8)

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x (image_size//16) x (image_size//16)

                # Reshape(-1, ndf * 8 * (image_size // 16) ** 2),
                Reshape(ndf * 8 * (image_size // 16) ** 2),
                nn.Linear(ndf * 8 * (image_size // 16) ** 2, 1),
                nn.Sigmoid()
            )

        def forward(self, img, c):
            output = self.emb(c)
            output = output.view(c.size(0), 1, image_size, image_size)
            output = torch.cat((img, output), 1)
            output = self.main(output)
            return output


    # -------------------------------------------------------------------------------------
    # 读取网络
    net_G = Generator(ngpu).to(device)
    net_D = Discriminator(ngpu).to(device)

    print("Loading checkpoint...")

    checkpoint = torch.load(ckpt_path)
    # sample_noise = checkpoint['sample_noise']

    net_D.load_state_dict(checkpoint['netD_state_dict'])
    net_G.load_state_dict(checkpoint['netG_state_dict'])

    net_D.eval()
    net_G.eval()

    # -------------------------------------------------------------------------------------
    # 生成样本
    # 初始化输入噪声与标签
    n_sample = 64  # 生成图片数量
    # manualSeed = 999    # Set random seem for reproducibility
    # torch.manual_seed(manualSeed)
    sample_noise = torch.randn(n_sample, latent_dim, device=device)
    sample_labels = torch.zeros(n_sample, dtype=torch.int64, device=device)

    # 噪声每n_classes次循环，标签每次循环
    # 废弃
    # sni = 0  # parameter for iteration, 后面的噪声与第sni个噪声相同
    # for i in range(n_sample):
    #     print(i, sni, i - i % n_classes)
    #     sample_labels[i] = torch.tensor(i % n_classes, dtype=torch.int64, device=device)
    #     sample_noise[i] = sample_noise[sni]
    # 等价于
    for i in range(n_sample):
        sample_labels[i] = torch.tensor(i % n_classes, dtype=torch.int64, device=device)
        sample_noise[i] = sample_noise[i - i % n_classes]

    # 转为one-hot
    sample_labels = torch.zeros(n_sample, n_classes, device=device).scatter(1, sample_labels.view(-1, 1), 1)

    samples = net_G(sample_noise, sample_labels)
    vutils.save_image(samples, os.path.join(samples_path, 'sample.jpg'), nrow=8, padding=2, normalize=True)
    # -------------------------------------------------------------------------------------
    # 噪声插值
    # 初始化输入噪声与标签
    n_sample = 8  # 生成图片数量
    # manualSeed = 999    # Set random seem for reproducibility
    # torch.manual_seed(manualSeed)
    sample_noise = torch.randn(n_sample, latent_dim, device=device)
    sample_labels = torch.zeros(n_sample, dtype=torch.int64, device=device)

    for i in range(n_sample):
        sample_noise[i] = torch.lerp(sample_noise[0], sample_noise[-1], i / (n_sample - 1))

    sample_labels = torch.zeros(n_sample, dtype=torch.int64, device=device)
    # sample_labels = torch.ones(n_sample, dtype=torch.int64, device=device)

    sample_labels = torch.zeros(n_sample, n_classes, device=device).scatter(1, sample_labels.view(-1, 1), 1)

    samples = net_G(sample_noise, sample_labels)
    vutils.save_image(samples, os.path.join(samples_path, 'noise_interpolation.jpg'), nrow=8, padding=2, normalize=True)

    # -------------------------------------------------------------------------------------
    # 标签插值
    n_sample = 8  # 生成图片数量
    # manualSeed = 999    # Set random seem for reproducibility
    # torch.manual_seed(manualSeed)

    # n_sample个相同的噪声
    sample_noise = torch.randn(1, latent_dim, device=device)
    sample_noise_temp = sample_noise
    for i in range(n_sample - 1):
        sample_noise = torch.cat((sample_noise, sample_noise_temp), 0)

    # 标签插值
    sample_labels = torch.zeros(n_sample, dtype=torch.int64, device=device)
    sample_labels[-1] = 1
    sample_labels = torch.zeros(n_sample, n_classes, device=device).scatter(1, sample_labels.view(-1, 1), 1)
    for i in range(n_sample):
        sample_labels[i] = torch.lerp(sample_labels[0], sample_labels[-1], i / (n_sample - 1))

    samples = net_G(sample_noise, sample_labels)
    vutils.save_image(samples, os.path.join(samples_path, 'label_interpolation.jpg'), nrow=8, padding=2, normalize=True)

    # -------------------------------------------------------------------------------------
    # 二维插值
    n_sample = 8  # 每边生成的图片数量
    # manualSeed = 999    # Set random seem for reproducibility
    # torch.manual_seed(manualSeed)

    # 标签插值
    sample_labels = torch.zeros(n_sample, dtype=torch.int64, device=device)
    sample_labels[-1] = 1
    sample_labels = torch.zeros(n_sample, n_classes, device=device).scatter(1, sample_labels.view(-1, 1), 1)
    for i in range(n_sample):
        sample_labels[i] = torch.lerp(sample_labels[0], sample_labels[-1], i / (n_sample - 1))

    # 标签维数扩张
    sample_labels_temp = sample_labels
    for i in range(n_sample - 1):
        sample_labels = torch.cat((sample_labels, sample_labels_temp), 0)

    # 噪声插值
    sample_noise_x = torch.randn(n_sample, latent_dim, device=device)
    for i in range(n_sample):
        sample_noise_x[i] = torch.lerp(sample_noise_x[0], sample_noise_x[-1], i / (n_sample - 1))

    sample_noise = torch.randn(n_sample ** 2, latent_dim, device=device)

    for i in range(n_sample ** 2):
        # print(i, i // n_sample)
        sample_noise[i] = sample_noise_x[i // n_sample]

    st = time.time()
    samples = net_G(sample_noise, sample_labels)
    ct = time.time() - st
    print('G_cost:\t%.3f'% ct)

    vutils.save_image(
        samples, os.path.join(samples_path, 'two_dim_interpolation.jpg'), nrow=8, padding=2, normalize=True
    )

    # # -------------------------------------------------------------------------------------
    # # 测试D
    # # one-hot 转 lable
    # sample_labels = sample_labels.argmax(1)
    # st = time.time()
    # pre = net_D(samples, sample_labels)
    # ct = time.time() - st
    #
    # pre = pre.mean()
    # print('D_cost:\t%.3f'% ct)
    #
    # print('pre_fake:\t', pre)
    # pre = net_D(real_img, real_c)
    # pre = pre.mean()
    # print('pre_real:\t', pre)
