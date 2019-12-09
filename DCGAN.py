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

    max_epochs = 1000  # Number of training epochs
    lr = 0.0002  # Learning rate for optimizers
    batch_size = 64  # Batch size during training
    image_size = 128  # All images will be resized to this size using a transformer.

    is_load = False
    ckpt_path = './DCGAN_Samples/checkpoint_iteration_150.tar'

    # Root directory for dataset
    data_path = '/datasets/celeba'
    samples_path = './DCGAN_Samples'
    os.makedirs(samples_path, exist_ok=True)

    latent_dim = 100  # Size of z latent vector (i.e. size of generator input)
    nc = 3  # Number of channels in the training images. For color images this is 3
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers

    ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
    workers = 0  # Number of workers for dataloader
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Set random seem for reproducibility
    # manualSeed = random.randint(1, 10000) # use if you want new results
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    manualSeed = 999
    torch.manual_seed(manualSeed)
    # manual_seed的作用期很短
    sample_noise = torch.randn(64, latent_dim, device=device)

    # Create the dataset
    dataset = dset.ImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)


    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

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


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x image_size x image_size
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
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

        def forward(self, input):
            output = self.main(input)
            return output


    net_G = Generator(ngpu).to(device)
    net_D = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net_G = nn.DataParallel(net_G, list(range(ngpu)))
        net_D = nn.DataParallel(net_D, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    net_G.apply(weights_init)
    net_D.apply(weights_init)

    print(net_G)
    print(net_D)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(beta1, 0.999))


    # define a method to save loss image

    def save_loss_image(list_loss_G, list_loss_D, path):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(list_loss_G, label="G")
        plt.plot(list_loss_D, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(path, 'loss.jpg'))
        plt.close()

    def save_accuracy_image(list_D_G_z, list_D_x, path):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator accuracy During Training")
        plt.plot(list_D_G_z, label="D(G(z))")
        plt.plot(list_D_x, label="D(x)")
        plt.xlabel("iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(path, 'Accuracy.jpg'))
        plt.close()


    # Training Loop
    # Lists to keep track of progress
    if is_load:
        print("Loading checkpoint...")

        checkpoint = torch.load(ckpt_path)
        last_epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        last_i = checkpoint['last_current_iteration']
        sample_noise = checkpoint['sample_noise']

        list_loss_D = checkpoint['list_loss_D']
        list_loss_G = checkpoint['list_loss_G']
        list_D_x = checkpoint['list_D_x']
        list_D_G_z = checkpoint['list_D_G_z']

        net_D.load_state_dict(checkpoint['netD_state_dict'])
        net_G.load_state_dict(checkpoint['netG_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        net_D.eval()
        net_G.eval()

    else:
        last_epoch = 0
        iteration = 0

        list_loss_G = []
        list_loss_D = []
        list_D_x = []
        list_D_G_z = []

    print("Starting Training Loop...")
    for epoch in range(last_epoch, max_epochs):
        # 若读取，重置当前周期，且只执行一次
        str_i = 0
        if is_load:
            str_i = last_i
            is_load = False
        for i, data in enumerate(dataloader, str_i):

            # -----------------------------------------------------------
            # Initial batch
            real = data[0].to(device)
            real_batch_size = real.size(0)
            ones = torch.full((real_batch_size, 1), 1, device=device)
            zeros = torch.full((real_batch_size, 1), 0, device=device)
            noise = torch.randn(real_batch_size, latent_dim, device=device)
            fake = net_G(noise)

            # -----------------------------------------------------------
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            net_D.zero_grad()

            # Calculate D loss
            # BCE(x,y) = - mean(y(log(x)) + (1-y)log(1-x))
            loss_real = criterion(net_D(real), ones)            # loss of real batch
            loss_fake = criterion(net_D(fake.detach()), zeros)            # loss of fake batch

            # 等价于
            # loss_real = - torch.mean(torch.log(netD(real)))
            # loss_fake = - torch.mean(torch.log(1 - netD(fake.detach())))

            loss_D = (loss_real + loss_fake) * 0.5            # total loss of D


            # Update D
            loss_D.backward()
            optimizer_D.step()

            # -----------------------------------------------------------
            # Update G network: maximize log(D(G(z)))
            net_G.zero_grad()

            # Calculate G loss
            loss_G = criterion(net_D(fake), ones)

            # 等价于
            # loss_G = - torch.mean(torch.log(netD(fake)))

            # Update G
            loss_G.backward()
            optimizer_G.step()

            # -----------------------------------------------------------
            # Output training stats
            with torch.no_grad():
                list_loss_G.append(loss_G.item())
                list_loss_D.append(loss_D.item())

                if i % 50 == 0:
                    D_x = net_D(real).mean().item()
                    D_G_z = net_D(fake).mean().item()
                    list_D_x.append(D_x)
                    list_D_G_z.append(D_G_z)

                    print(
                        '[%d/%d][%2d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %5.2f%%\tD(G(z)): %5.2f%%'
                        % (epoch, max_epochs, i, len(dataloader),
                           loss_D.item(), loss_G.item(), D_x * 100, D_G_z * 100))

                # Check how the generator is doing by saving G's output on sample_noise
                if (iteration % 100 == 0) or ((epoch == max_epochs - 1) and (i == len(dataloader) - 1)):
                    # with torch.no_grad():
                    #     sample = netG(fixed_noise).detach().cpu()
                    samples = net_G(sample_noise).cpu()
                    vutils.save_image(samples, os.path.join(samples_path, '%d.jpg' % iteration), padding=2,
                                      normalize=True)
                    save_loss_image(list_loss_G, list_loss_D, samples_path)
                    save_accuracy_image(list_D_G_z, list_D_x, samples_path)

                # Save model
                if (iteration % 1000 == 0) or ((epoch == max_epochs - 1) and (i == len(dataloader) - 1)):
                    save_path = os.path.join(samples_path, 'checkpoint_iteration_%d.tar' % iteration)
                    torch.save({
                        'epoch': epoch,
                        'iteration': iteration,
                        'last_current_iteration': i,
                        'netD_state_dict': net_D.state_dict(),
                        'netG_state_dict': net_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'loss_D': loss_D,
                        'loss_G': loss_G,
                        'list_loss_D': list_loss_D,
                        'list_loss_G': list_loss_G,
                        'list_D_x': list_D_x,
                        'list_D_G_z': list_D_G_z,
                        'sample_noise': sample_noise
                    }, save_path)

            # iteration: total iteration, i: iteration of current epoch
            iteration += 1
