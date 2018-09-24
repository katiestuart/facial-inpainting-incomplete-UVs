from __future__ import division

# Import libraries
import os
from random import randint
import matplotlib.pyplot as plt
import numpy as np

import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from torch.nn import functional as F
from tqdm import tqdm


def random_box_inbb(img, size=10, display_yn=False, perc_zero_cutoff=0.1,box_color='white'):
    """
        Find a box with non-zero pixels (Main image), this represents inside main face

            1) find the maximum box surrounding the face
            2) Take a random box and check non-zero rate
            3) Nostrils and eyes also have zeros so we have to apply a threshold for acceptance
            4) If no box can be found, random box is taken    i.e '006319.png'
            5) create a black and white mask of the box to apply to loss function
    """
    # some images are completely black and fail  i.e '006597.png'
    if box_color == 'white':
        nonbox_color = 'black'
    elif box_color == 'black':
        nonbox_color = 'white'

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
                img_draw.rectangle((left_rnd, upper_rnd, left_rnd+size, upper_rnd+size), outline=box_color, fill=box_color)
                # img_draw.rectangle((left_rnd, upper_rnd, left_rnd+size, upper_rnd+size), outline='white', fill='white') #original
                error_score = 0

        # store error images indicator
        if perc_zero > perc_zero_cutoff:
            error_score = 1
        else:
            error_score = 0

        # Create the black and white mask
        img_draw_mask.rectangle((0, 0, 127, 127), outline=nonbox_color, fill=nonbox_color)
        # img_draw_mask.rectangle((0, 0, 127, 127), outline='black', fill='black') #original
        img_draw_mask.rectangle((left_rnd, upper_rnd, left_rnd+size, upper_rnd+size), outline=box_color, fill=box_color)
        # img_draw_mask.rectangle((left_rnd, upper_rnd, left_rnd+size, upper_rnd+size), outline='white', fill='white') #original
        if display_yn == True:
            print('\n'+'*'*40+'\nimg_copy')
            display(img_copy)
            print('\n'+'*'*40+'\nimg_copy2')
            display(img_mask)
        return img_copy, img_mask, error_score
    except (ValueError,TypeError) as e:
        # Show the black image
        return img, img, 1

######################################
# Run through and save BOX & BOXMASK
######################################
resample_images='Y'
for i in range(1):
    if resample_images=='Y':
        box_im_PATH = project_path+'box_images/'
        box_mask_PATH = project_path+'box_masks/'
        counter=0
        error_images=[]
        for a in df.image_name.values:
            counter+=1
            print(counter)
            a_img = Image.open(IMG_PATH + a)
            a_new, a_mask, error_scoreA = random_box_inbb(a_img, size=30, display_yn=False, perc_zero_cutoff=0.20, box_color='black')

            if error_scoreA==1:
                # error images
                print('ERROR IMAGE:',a)
                display(a_new)
                error_images.append(a)
                a_new.save(box_im_PATH+a)
                a_mask.save(box_mask_PATH+a)
            else:
                # save non error images
                a_new.save(box_im_PATH+a)
                a_mask.save(box_mask_PATH+a)
        print('Boxes and Masked resampled and saved')
    else:
        print('Boxes and Masked on file used')

Save error images
pd.DataFrame(error_images).to_csv(project_path+'box_error_images.csv',index=None)


class UVDataset(Dataset):
    """
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, box_path, mask_path, transform=None, train=True):

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

        self.train = train
        self.transform = transform
        self.dfc = tmp_df

        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['tags'].astype(int)

        # Dummy up an array of random values
        self.y_train = [np.ndarray(x) for x in self.y_train]

    def __getitem__(self, index):
        transform_tensor = transforms.Compose([transforms.ToTensor()])
        img = Image.open(self.img_path + self.X_train[index])
        img = img.convert('RGB')
        img_box = Image.open(self.box_path + self.X_train[index])
        img_box = img_box.convert('RGB')
        img_mask = Image.open(self.mask_path + self.X_train[index])
        img_mask = img_mask.convert('RGB')


        if self.transform is not None:
            img = self.transform(img)
            img_box = self.transform(img_box)
            img_mask = transform_tensor(img_mask)

        #label = torch.from_numpy(self.y_train[index])
        label = self.X_train[index]

        return img, label, img_box, img_mask

    def __len__(self):
        return len(self.X_train.index)
