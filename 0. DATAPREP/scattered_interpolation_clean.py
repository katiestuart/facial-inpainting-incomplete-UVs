# Import libraries

import os
from __future__ import division
from random import randint
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
#import torchvision.datasets as dset
from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from torch.nn import functional as F

from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from matplotlib.mlab import griddata
from scipy import interpolate

################################
# PATH DEFINITIONS
################################

project_path = 'path/'

df=pd.read_csv(project_path+'/image_names_test_train.csv')
IMG_PATH_IMG = project_path+'images/'
IMG_PATH = project_path+'tex/'
IMG_EXT = '.jpg'
h_PATH = project_path+'h/'
i_PATH = project_path+'images/'
v_PATH = project_path+'v/'
img_path_inpaint = project_path+'tex_inpaint/'
IMG_EXT = '.jpg'
tex = project_path+'tex_new/'
box_im_PATH = project_path+'box_images/'
box_mask_PATH = project_path+'box_masks/'


from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(IMG_PATH_IMG) if isfile(join(IMG_PATH_IMG, f))]
for i in onlyfiles[:]:
    if not(i.endswith(".png")):
        onlyfiles.remove(i)

type(onlyfiles)


####### New Test ##########
from scipy.ndimage import rotate

counter=0
for a in onlyfiles:
    try:
        counter+=1
        # a = onlyfiles[0]
        # load img, h,v coords
        h = Image.open(h_PATH + a)
        v = Image.open(v_PATH + a)
        i = Image.open(i_PATH + a)

        h_np = np.array(h)
        v_np = np.array(v)
        hArray = h_np.reshape(1,-1)
        hArray = 360 - hArray
        vArray = v_np.reshape(1,-1)

        numIndexes = 128
        hi = np.linspace(np.min(hArray), np.max(hArray),numIndexes)
        vi = np.linspace(np.min(vArray), np.max(vArray),numIndexes)

        HI, VI = np.meshgrid(hi, vi)
        points = np.vstack((hArray,vArray)).T

        # 3 channels
        r,g,b = i.split()
        r_np = np.array(r)
        g_np = np.array(g)
        b_np = np.array(b)

        r_np = r_np.reshape(1,-1)
        g_np = g_np.reshape(1,-1)
        b_np = b_np.reshape(1,-1)

        values = np.asarray(r_np)
        values = values.reshape(-1,1)

        DEM = interpolate.griddata(points, values, (HI,VI), method=int_method,rescale=int_rescale)

        values = np.asarray(g_np)
        values = values.reshape(-1,1)
        DEM_1 = interpolate.griddata(points, values, (HI,VI), method=int_method,rescale=int_rescale)

        values = np.asarray(b_np)
        values = values.reshape(-1,1)
        DEM_2 = interpolate.griddata(points, values, (HI,VI), method=int_method,rescale=int_rescale)

        DEM_IMG = Image.fromarray(DEM[:,:,0]).convert('L')
        DEM_IMG_1 = Image.fromarray(DEM_1[:,:,0]).convert('L')
        DEM_IMG_2 = Image.fromarray(DEM_2[:,:,0]).convert('L')

        out = Image.merge("RGB", (DEM_IMG, DEM_IMG_1, DEM_IMG_2))
        out = rotate(out, 180, reshape=False)
        out_img = Image.fromarray(out,'RGB')
        out_img.save(tex+a)
    except:
        pass


#### Test Interpolation Method

#
# h = Image.open(h_PATH + '004670.png')
# v = Image.open(v_PATH + '004670.png')
# i = Image.open(i_PATH + '004670.png')
#
# h_np = np.array(h)
# v_np = np.array(v)
# i_np = np.array(i)
# i_np_0 = i_np[:,:,0]
#
#
# hArray = h_np.reshape(1,-1)
# hArray = 360 - hArray
# vArray = v_np.reshape(1,-1)
# iArray = i_np_0.reshape(1,-1)
#
# iArray.shape
#
# numIndexes = 128
# hi = np.linspace(np.min(hArray), np.max(hArray),numIndexes)
# vi = np.linspace(np.min(vArray), np.max(vArray),numIndexes)
#
# HI, VI = np.meshgrid(hi, vi)
# points = np.vstack((hArray,vArray)).T
#
# c = Image.open(i_PATH + '004670.png')
# plt.imshow(c)
# plt.show()
#
# c = Image.open(IMG_PATH + '004670.png')
# plt.imshow(c)
# plt.show()
#
# #METHOD
# #int_method = 'linear'
# # int_method = 'nearest'
# int_method = 'cubic'
#
# int_rescale=False
#
# # 3 channels
# r,g,b = i.split()
# r_np = np.array(r)
# g_np = np.array(g)
# b_np = np.array(b)
#
#
# r_np = r_np.reshape(1,-1)
# g_np = g_np.reshape(1,-1)
# b_np = b_np.reshape(1,-1)
#
# values = np.asarray(r_np)
# values = values.reshape(-1,1)
#
# values.shape
# #values = np.flip(values,0)
#
# DEM = interpolate.griddata(points, values, (HI,VI), method=int_method,rescale=int_rescale)
#
# values = np.asarray(g_np)
# values = values.reshape(-1,1)
# DEM_1 = interpolate.griddata(points, values, (HI,VI), method=int_method,rescale=int_rescale)
#
# values = np.asarray(b_np)
# values = values.reshape(-1,1)
# DEM_2 = interpolate.griddata(points, values, (HI,VI), method=int_method,rescale=int_rescale)
#
# DEM_IMG = Image.fromarray(DEM[:,:,0]).convert('L')
# DEM_IMG_1 = Image.fromarray(DEM_1[:,:,0]).convert('L')
# DEM_IMG_2 = Image.fromarray(DEM_2[:,:,0]).convert('L')
# # plt.imshow(DEM_IMG)
# # plt.show()
#
# out = Image.merge("RGB", (DEM_IMG, DEM_IMG_1, DEM_IMG_2))
# from scipy.ndimage import rotate
# rot = rotate(out, 180, reshape=False)
#
#
# out_img = Image.fromarray(rot,'RGB')
# out_img
#
# # INTERPOLATION
# plt.imshow(rot)
# plt.show()
#
# # ACTUAL
# c = Image.open(IMG_PATH + '004670.png')
# plt.imshow(c)
# plt.show()
