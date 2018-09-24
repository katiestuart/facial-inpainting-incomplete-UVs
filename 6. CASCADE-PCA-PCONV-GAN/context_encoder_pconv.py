import torch
import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
import project_data_pconv as p
from models_full_image import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# add GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str,default=False)
args = parser.parse_args()

print('*'*50)
print('--- ARGS ---')
print('gpu',args.gpu)
print('gpu',type(args.gpu))
print('*'*50)




cwd = os.getcwd()
if 'ubuntu' in cwd:
    project_path = os.getcwd()+'/'
    data_path = '/home/ubuntu/data/'
    print('ProjectData project_path = ',cwd)
else:
    project_path = 'path'


gan_path =cwd
IMG_PATH = data_path+'tex/'

# reload saved - keep same test / train split
df=pd.read_csv(data_path+'/image_names_test_train_sample.csv')
csv = data_path+'/image_names_test_train_sample.csv'
df_train = df[df.train==True]
df_test = df[df.train==False]

# os.makedirs('images', exist_ok=True)
# parser = argparse.ArgumentParser()
# parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
# parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
# parser.add_argument('--dataset_name', type=str, default='img_align_celeba', help='name of the dataset')
# parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
# parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
# parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
# parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
# parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
# parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
# parser.add_argument('--mask_size', type=int, default=64, help='size of random mask')
# parser.add_argument('--channels', type=int, default=3, help='number of image channels')
# parser.add_argument('--sample_interval', type=int, default=500, help='interval between image sampling')
# opt = parser.parse_args()
# print(opt)

n_epochs=50000
batch_size=16
dataset_name='img_align_celeba'
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 4
latent_dim = 100
img_size = 128
mask_size = 32
channels = 3
sample_interval = 10

# cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
# patch_h, patch_w = int(mask_size / 2**3), int(mask_size / 2**3)
# patch = (1, patch_h, patch_w)
# patch = (1, 8, 8)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############
# GPU
##############
if args.gpu=='True':
    device = torch.device('cuda')
    print('GPU DEVICE')


# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()


if args.gpu=='True':
    # Initialize generator and discriminator
    generator = Generator(channels=channels).to(device)
    discriminator = Discriminator(channels=channels).to(device)
    # Loss function
    adversarial_loss = torch.nn.MSELoss().to(device)
    pixelwise_loss = torch.nn.L1Loss().to(device)

else:
    # Initialize generator and discriminator
    generator = Generator(channels=channels)
    discriminator = Discriminator(channels=channels)
    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    pixelwise_loss = torch.nn.L1Loss()


# if cuda:
#     generator.cuda()
#     discriminator.cuda()
#     adversarial_loss.cuda()
#     pixelwise_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)



# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


def save_sample(batches_done, test_dataloader):
    import os
    if os.path.isdir('images/')==False:
        os.mkdir('images/')
    #coord_u, coord_l = p.remask('Y', df=df_test)
    #train, test = p.data_gen(norm='Y')

    # test_dataloader = DataLoader(test, batch_size=1, shuffle=True)
    test_dataloader = iter(DataLoader(test, batch_size=1, shuffle=True))

    counter=0
    # for i in test_dataloader:
    for i in range(100):
        try:
            data_it = next(test_dataloader)
            counter+=1
            if counter <13:
                # if args.gpu=='True':
                #     imgs, labels, img_box, img_mask, pconv_img = [x.to(device) for x in next(iter(test_dataloader))]
                # else:

                # imgs, labels, img_box, img_mask, pconv_img = next(iter(test_dataloader))
                imgs, labels, img_box, img_mask, pconv_img = data_it
                imgs = imgs.cuda()
                # labels = labels.cuda()
                img_box = img_box.cuda()
                img_mask = img_box.cuda()
                pconv_img = pconv_img.cuda()



                print('Sample Ran', imgs.size())
                imgs = Variable(imgs.type(torch.Tensor))
                pconv_img = Variable(pconv_img.type(torch.Tensor))

                x_unorm = imgs*0.5+0.5
                full_mask = custom_replace(x_unorm, 0, 1)

                if args.gpu=='True':
                    imgs = imgs.cuda()
                    pconv_img = pconv_img.cuda()

                generator.eval()
                gen_img = generator(pconv_img)

                if args.gpu=='True':
                    full_mask = full_mask.cuda()
                    gen_img = gen_img.cuda()

                test_loss = pixelwise_loss(gen_img*full_mask, imgs*full_mask)

                try:
                    gen_img_concat
                    gen_img_concat = torch.cat((gen_img_concat, gen_img.data))
                    #test_plot_concat = torch.cat((test_plot_concat, filled_samples.data))
                    imgs_concat = torch.cat((imgs_concat, imgs.data))
                except:
                    gen_img_concat = gen_img.data
                    #test_plot_concat = filled_samples.data
                    imgs_concat = imgs.data
        except FileNotFoundError:
            print('Missing Image',labels )
        except:
            print('Another error')



    # Save sample
    sample = torch.cat((gen_img_concat, imgs_concat), -2)
    save_image(sample,'images/%d.png' % batches_done, nrow=6, normalize=True)

    return test_loss



def custom_replace(tensor, on_zero, on_non_zero):
    """ Create Tensor with binary values
        Rule: Non Zero
    """
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res


def show_image(img, norm='Y'):
    img = img.view(3,128,128)
    # revert the normalisation for displaying the images in their original form
    if norm == 'Y':
        img = img * 0.5 + 0.5
    # Convert to numpy for visualisation
    npimg = img.detach().numpy()
    # Plot each image
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



# for epoch in range(1):
train_losses_g = []
train_losses_d = []
test_losses = []
pixel_loss = []
for epoch in range(n_epochs):
    #coord_u, coord_l = p.remask('N', df=df_train)
    #train, test = p.data_gen(norm='Y', coords_u=coord_u, coords_l=coord_l)
    train, test = p.data_gen(norm='Y')
    dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=12, shuffle=True)
    counter=0
    for i in dataloader:
        counter+=1


        imgs, labels, masked_imgs, box_mask, pconv = i
        x_unorm = imgs*0.5+0.5
        full_mask = custom_replace(x_unorm, 0, 1)
        # print('fm:', full_mask.shape)
        img_sum = torch.sum(full_mask,1)
        # print('img_sum:', img_sum.shape)
        valid = custom_replace(img_sum, 0, 1)
        # valid = valid.reshape(batch_size, -1, imgs.shape[2], imgs.shape[3])
        valid = valid.reshape(imgs.shape[0], -1, imgs.shape[2], imgs.shape[3])


        fake = Variable(torch.Tensor(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]).fill_(0.0), requires_grad=False)

        # Configure input
        imgs = Variable(imgs.type(torch.Tensor))
        pconv = Variable(pconv.type(torch.Tensor))


        if args.gpu=='True':
            valid = valid.cuda()
            fake = fake.cuda()
            imgs = imgs.cuda()
            full_mask = full_mask.cuda()
            pconv = pconv.cuda()


        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        generator.train()
        gen_parts = generator(pconv)

        g_adv = adversarial_loss(discriminator(gen_parts*full_mask), valid)

        g_pixel = pixelwise_loss(gen_parts*full_mask, imgs*full_mask)


        # Total loss
        g_loss = 0.001 * g_adv + 0.999 * g_pixel

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(imgs*full_mask), valid)

        fake_loss = adversarial_loss(discriminator((gen_parts.detach())*full_mask), fake) ##### take out full mask?
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()


        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]" % (epoch, n_epochs, counter, len(dataloader),
                                                            d_loss.item(), g_adv.item(), g_pixel.item()))

        # print('*'*50)

    # Generate sample at sample interval
    if os.path.isdir(gan_path+'/models/')==False:
        os.mkdir(gan_path+'/models/')
    print('Saved loss plots in:',gan_path+'models/')
    batches_done = epoch * len(dataloader) + counter
    if batches_done % sample_interval == 0:
        print('saving_sample')
        test_loss = save_sample(epoch, test_dataloader)
        test_losses.append(test_loss)


    train_losses_g.append(g_loss)
    train_losses_d.append(d_loss)
    pixel_loss.append(g_pixel)
    train_plot = p.two_plots(train_losses_g, 'generator loss', train_losses_d, 'discriminator loss', 'epoch',
                            title = 'gen vs. dis', save_name=gan_path+'models/'+'/train_loss_plt.png',show_plot=False)

    test_plot = p.two_plots(pixel_loss, 'train loss', test_losses, 'test loss', 'epoch',
                            title = 'train vs. test', save_name=gan_path+'models/'+'/test_loss_plt.png',show_plot=False)
