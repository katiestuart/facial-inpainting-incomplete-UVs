# Import libraries
from __future__ import division
import os
from random import randint
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw
from torch.nn import functional as F
import matplotlib.pyplot as plt

# import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
################################
# PATH DEFINITIONS
################################

cwd = os.getcwd()
if 'ubuntu' in cwd:
    project_path = os.getcwd()+'/'
    data_path = '/home/ubuntu/data/'
    print('ProjectData project_path = ',cwd)
else:
    project_path = 'path'

IMG_PATH = data_path+'tex/'
PCONV_PATH = '/home/ubuntu/data/PCONV-pca-in-dilation-GPU-PCA32/models/output_imgs/'
IMG_EXT = '.jpg'

h_PATH = data_path+'h/'
i_PATH = data_path+'images/'
v_PATH = data_path+'v/'
img_path_inpaint = data_path+'tex_inpaint/'
IMG_EXT = '.jpg'

box_im_PATH = data_path+'box_images/'
box_mask_PATH = data_path+'box_masks/'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))]
for i in onlyfiles[:]:
    if not(i.endswith(".png")):
        onlyfiles.remove(i)

# reload saved - keep same test / train split
df=pd.read_csv(data_path+'/image_names_test_train_sample.csv')
csv = data_path+'/image_names_test_train_sample.csv'



################################
# MODEL PREP
################################

def two_plots(p1_data, p1y_label, p2_data, p2y_label, x_label, title, save_name, show_plot=True):
    try:
        plt.close(fig)
    except:
        pass
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(x_label)

    ax1.set_ylabel(p1y_label, color=color)
    ax1.plot(p1_data, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(p2y_label, color=color)  # we already handled the x-label with ax1
    ax2.plot(p2_data, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle(title)
    if show_plot==True:
        plt.show()
    fig.savefig(save_name)


def img_to_numpy(img):
    transform_tensor = transforms.Compose([transforms.ToTensor()])
    img1_ = transform_tensor(img)
    img1_ = img1_.numpy()
    return img1_



def random_box_inbb(img, size=30, display_yn=False, perc_zero_cutoff=0.2):
    """
        Find a box with non-zero pixels (Main image), this represents inside main face

            1) find the maximum box surrounding the face
            2) Take a random box and check non-zero rate
            3) Nostrils and eyes also have zeros so we have to apply a threshold for acceptance
            4) If no box can be found, random box is taken    i.e '006319.png'
            5) create a black and white mask of the box to apply to loss function
    """
    # some images are completely black and fail  i.e '006597.png'
    try:
        left = img.getbbox()[0]
        upper = img.getbbox()[1]
        right = img.getbbox()[2]
        lower = img.getbbox()[3]

        cropped_im_min = 0
        perc_zero = 1

        if display_yn==True:
            print('\n'+'*'*40+'\nOriginal Image')
            display(img)
        counterI =0
        # trys 10 times then just accept that cant find box that meets the criteria
        while counterI < 10:
            counterI+=1

            if perc_zero > perc_zero_cutoff:
                img_copy = img.copy()
                img_mask = img.copy()

                img_draw = ImageDraw.Draw(img_copy)
                img_draw_mask = ImageDraw.Draw(img_mask)

                left_rnd = randint(left, right-size)
                upper_rnd = randint(upper, lower-size)

                cropped_im = img_copy.crop((left_rnd, upper_rnd, left_rnd+size, upper_rnd+size))
                cropped_im_np = img_to_numpy(cropped_im)
                cropped_im_min = cropped_im_np.min()
                non_zero = np.count_nonzero(cropped_im_np)
                perc_zero = 1-(non_zero/(cropped_im_np.shape[0]*cropped_im_np.shape[1]*cropped_im_np.shape[2]))

                if display_yn == True:
                    print('\n'+'*'*40+'\nCropped Image')
                    display(cropped_im)
                    if perc_zero > perc_zero_cutoff:
                        print('WARNING contains more than %s Zeros:' %perc_zero_cutoff ,perc_zero)
                    else:
                        print('Not more than %s Zeros:' %perc_zero_cutoff,perc_zero)

                ## THIS OVERWRITES THE ORIGINAL IMG
                img_draw.rectangle((left_rnd, upper_rnd, left_rnd+size, upper_rnd+size), outline='black', fill='black')
                error_score = 0

        # store error images indicator
        if perc_zero > perc_zero_cutoff:
            error_score = 1
        else:
            error_score = 0

        # Create the black and white mask
        img_draw_mask.rectangle((0, 0, 127, 127), outline='white', fill='white')
        img_draw_mask.rectangle((left_rnd, upper_rnd, left_rnd+size, upper_rnd+size), outline='black', fill='black')
        if display_yn == True:
            print('\n'+'*'*40+'\nimg_copy')
            display(img_copy)
            print('\n'+'*'*40+'\nimg_copy2')
            display(img_mask)
        return img_copy, img_mask, error_score
    except (ValueError,TypeError) as e:
        # Show the black image
        return img, img, 1


def custom_replace(tensor, on_zero, on_non_zero):
    """ Create Tensor with binary values
        Rule: Non Zero
    """
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

######################################
# Run through and save BOX & BOXMASK
######################################
# if not os.path.exists(project_path+'box_images/'):
#     os.makedirs(project_path+'box_images/')
#     os.makedirs(project_path+'box_masks/')
#
# resample_images='Y'
# for i in range(1):
#     if resample_images=='Y':
#         box_im_PATH = project_path+'box_images/'
#         box_mask_PATH = project_path+'box_masks/'
#         counter=0
#         error_images=[]
#         for a in df.image_name.values:
#             counter+=1
#             print(counter)
#             a_img = Image.open(IMG_PATH + a)
#             a_new, a_mask, error_scoreA = random_box_inbb(a_img, size=30, display_yn=False, perc_zero_cutoff=0.20)
#             #print(type(a_new))
#             if error_scoreA==1:
#                 # error images
#                 print('ERROR IMAGE:',a)
#                 #print(type(a_new))
#                 error_images.append(a)
#             else:
#                 # save non error images
#                 a_new.save(box_im_PATH+a)
#                 a_mask.save(box_mask_PATH+a)
#                 #print(type(a_new))
#         print('Boxes and Masked resampled and saved')
#     else:
#         print('Boxes and Masked on file used')

# img_tf = transforms.Compose(
#     [transforms.Resize(size=size), transforms.ToTensor(),
#      transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
#
# mask_tf = transforms.Compose(
#     [transforms.Resize(size=size), transforms.ToTensor()])


transform_tensor = transforms.Compose([transforms.ToTensor()])

class UVDataset(Dataset):
    """
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, box_path, mask_path, PCONV_PATH, transform=None, train=True):

        tmp_df = pd.read_csv(csv_path)
        if train==True:
            tmp_df=tmp_df[tmp_df.train==True]
            tmp_df['tags']=1
            tmp_df = tmp_df.reset_index(drop=True)

        else:
            tmp_df=tmp_df[tmp_df.train==False]
            tmp_df['tags']=2
            tmp_df = tmp_df.reset_index(drop=True)

        self.img_path = img_path
        self.box_path = box_path
        self.mask_path = mask_path
        self.pconv_path = PCONV_PATH

        self.train = train
        self.transform = transform
        self.dfc = tmp_df

        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['tags'].astype(int)

        # Dummy up an array of random values
        self.y_train = [np.ndarray(x) for x in self.y_train]

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index])
        img = img.convert('RGB')
        img_box = Image.open(self.box_path + self.X_train[index])
        img_box = img_box.convert('RGB')
        img_mask = Image.open(self.mask_path + self.X_train[index])
        img_mask = img_mask.convert('RGB')
        pconv_img = Image.open(self.pconv_path + self.X_train[index])
        pconv_img = pconv_img.convert('RGB')


        if self.transform is not None:
            img = self.transform(img)
            img_box = self.transform(img_box)
            pconv_img = self.transform(pconv_img)

        transform_nnorm = transforms.Compose([transforms.ToTensor()])
        img_mask = transform_nnorm(img_mask)

        label = torch.from_numpy(self.y_train[index])
        label = self.X_train[index]

        return img, label, img_box, img_mask, pconv_img

    def __len__(self):
        return len(self.X_train.index)



############################
# Create train set loaders
############################

# Import dataset
batch_size = 128

# Convert the data to Tensor and normalise by the mean and std  of the data set
transform1 = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


transform1 = transforms.Compose([transforms.ToTensor()])

# No normalisation
transform2 = transforms.Compose([transforms.ToTensor()])

def data_gen(norm):
    norm = norm
    # New Box + Mask
    for i in range(1):
        if norm == 'Y':
            train_set = UVDataset(csv,IMG_PATH, box_im_PATH, box_mask_PATH, PCONV_PATH, transform=transform1, train=True)
            test_set = UVDataset(csv, IMG_PATH, box_im_PATH, box_mask_PATH, PCONV_PATH, transform=transform1, train=False)
            print('Normalised')
        # Original
        else:
            train_set = UVDataset(csv,IMG_PATH, box_im_PATH, box_mask_PATH, PCONV_PATH, transform=transform2, train=True)
            test_set = UVDataset(csv, IMG_PATH, box_im_PATH, box_mask_PATH, PCONV_PATH, transform=transform2, train=False)

            print('Not Normalised')
    return train_set, test_set
