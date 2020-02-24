# Import libraries
from __future__ import division
import os
from random import randint
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
# import opt
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

## import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
################################
# PATH DEFINITIONS
################################


if os.getcwd()=='/home/ubuntu/data':
    project_path = '/home/ubuntu/data/'
else:
    project_path = 'path'


IMG_PATH = project_path+'tex/'
PCA_PATH = project_path+'PCA/pca_128_full_output_images/'
PConv_PATH = project_path+'/data/PCONV-19k/models/output_imgs/'
IMG_EXT = '.jpg'



from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))]
for i in onlyfiles[:]:
    if not(i.endswith(".png")):
        onlyfiles.remove(i)

# reload saved - keep same test / train split
df=pd.read_csv(project_path+'/image_names_test_train_sample.csv')
csv = project_path+'/image_names_test_train_sample.csv'



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



def custom_replace(tensor, on_zero, on_non_zero):
    """ Create Tensor with binary values
        Rule: Non Zero
    """
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res


transform_tensor = transforms.Compose([transforms.ToTensor()])

class UVDataset(Dataset):
    """
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, PCA_PATH, Pconv_path, transform=None, train=True):

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

        self.pca_path = PCA_PATH
        self.pconv_path = Pconv_path

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
        pca_img = Image.open(self.pca_path + self.X_train[index])
        pca_img = pca_img.convert('RGB')
        pconv_img = Image.open(self.pconv_path + self.X_train[index])
        pconv_img = pconv_img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            # img_box = self.transform(img_box)
            pca_img = self.transform(pca_img)
            pconv_img = self.transform(pconv_img)

        label = torch.from_numpy(self.y_train[index])

        return img, label, pca_img, pconv_img

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
size = (128, 128)


transform1 = transforms.Compose([transforms.ToTensor()])


# No normalisation
transform2 = transforms.Compose([transforms.ToTensor()])

def data_gen(norm):
    norm = norm
    # New Box + Mask
    for i in range(1):
        if norm == 'Y':
            train_set = UVDataset(csv,IMG_PATH, PCA_PATH, PConv_PATH, transform=transform1, train=True)
            test_set = UVDataset(csv, IMG_PATH, PCA_PATH, PConv_PATH, transform=transform1, train=False)
            print('Normalised')
        # Original
        else:
            train_set = UVDataset(csv,IMG_PATH, PCA_PATH, PConv_PATH, transform=transform2, train=True)
            test_set = UVDataset(csv, IMG_PATH, PCA_PATH, PConv_PATH, transform=transform2, train=False)

            print('Not Normalised')
    return train_set, test_set
