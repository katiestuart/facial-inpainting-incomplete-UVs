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

print(torch.__version__)
parser = argparse.ArgumentParser()
parser.add_argument('--NOTES', type=str,default='no notes')
parser.add_argument('--max_iter', type=int, default=5000)
parser.add_argument('--vis_interval', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--save_images_interval', type=int, default=1000)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--num_hidden_neurons', type=int, default=64)
args = parser.parse_args()
device = torch.device('cuda')

print('Code now GPU enabled with CUDA')

def get_state_dict_on_cpu(obj):
    #cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key]
    return state_dict

def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']


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


if os.getcwd()=='/home/ubuntu/data':
    project_path = '/home/ubuntu/data/'
    pca_path = '/home/ubuntu/data/pca/'
else:
    project_path = '/path/'


IMG_PATH_SAMPLE = project_path+'tex/' #Sample 600
IMG_PATH = project_path+'tex_new/' #full 19k
IMG_EXT = '.jpg'
h_PATH = project_path+'h/'
i_PATH = project_path+'images/'
v_PATH = project_path+'v/'
img_path_inpaint = project_path+'tex_inpaint/'
IMG_EXT = '.jpg'

box_im_PATH = project_path+'box_images/'
box_mask_PATH = project_path+'custom_masks/'
box_mask_PATH_tex = project_path+'tex_masks/'


from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))]
for i in onlyfiles[:]:
    if not(i.endswith(".png")):
        onlyfiles.remove(i)



# reload saved - keep same test / train split
df=pd.read_csv(project_path+'/image_names_test_train.csv')
# df=pd.read_csv(project_path+'/image_names_test_train_sample.csv')
print('loaded',len(df),'images')

csv_sample = project_path+'/image_names_test_train_sample.csv'
csv = project_path+'/image_names_test_train.csv'

def img_to_numpy(img):
    transform_tensor = transforms.Compose([transforms.ToTensor()])
    img1_ = transform_tensor(img)
    img1_ = img1_.numpy()
    return img1_


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


## Check
# a = df.image_name.values[541]
# a_img = Image.open(IMG_PATH + '004670.png')
# img = a_img
# _new, a_mask, error_scoreA = random_box_inbb(a_img, size=30, display_yn=True, perc_zero_cutoff=0.20)


#####################################
Run through and save BOX & BOXMASK
#####################################
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


## Save error images
pd.DataFrame(error_images).to_csv(project_path+'box_error_images.csv',index=None)


transform_tensor = transforms.Compose([transforms.ToTensor()])
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



############################
# Create train set loaders
############################

# Import dataset
batch_size = args.batch_size

# Convert the data to Tensor and normalise by the mean and std  of the data set
transform1 = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

# No normalisation
transform2 = transforms.Compose([transforms.ToTensor()])

norm = 'Y'

# New Box + Mask
for i in range(1):
    if norm == 'Y':
        train_set = UVDataset(csv,IMG_PATH, box_im_PATH, box_mask_PATH, transform=transform1, train=True)
        test_set = UVDataset(csv, IMG_PATH, box_im_PATH, box_mask_PATH, transform=transform1, train=False)
        tex_set = UVDataset(csv_sample, IMG_PATH_SAMPLE, box_im_PATH, box_mask_PATH_tex, transform=transform1, train=False)
        print('Normalised')
    # Original
    else:
        train_set = UVDataset(csv,IMG_PATH, box_im_PATH, box_mask_PATH, transform=transform2, train=True)
        test_set = UVDataset(csv, IMG_PATH, box_im_PATH, box_mask_PATH, transform=transform2, train=False)
        tex_set = UVDataset(csv_sample, IMG_PATH_SAMPLE, box_im_PATH, box_mask_PATH_tex, transform=transform2, train=False)

        # train_set_nnorm = UVDataset(csv,IMG_PATH, box_im_PATH, box_mask_PATH, transform=transform1, train=True)
        # test_set_nnorm = UVDataset(csv,IMG_PATH, box_im_PATH, box_mask_PATH, transform=transform1, train=False)
        print('Not Normalised')


train_set_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=args.n_threads)

test_set_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.n_threads)

tex_set_loader = torch.utils.data.DataLoader(tex_set,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.n_threads)


# Load a batch of training images for visualising
data_iterator = iter(train_set_loader)
train_images, train_labels, train_boxes, train_masks = data_iterator.next()

data_iterator_test = iter(test_set_loader)
test_images, test_labels, test_boxes, train_masks = data_iterator_test.next()

data_iterator_tex = iter(tex_set_loader)
tex_images, tex_labels, tex_boxes, tex_masks = data_iterator_tex.next()


def custom_replace(tensor, on_zero, on_non_zero):
    """ Create Tensor with binary values
        Rule: Non Zero
    """
    if float(tensor.min())<0:
        tensor = tensor * 0.5 + 0.5
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res


def custom_replace_sumdim(tensor, on_zero, on_non_zero):
    """ Create Tensor with binary values
        Sums across all 3 channels
        inputs = [n_imgs, 3, H, W]
        Rule: Non Zero
    """
    if float(tensor.min())<0:
        tensor = tensor * 0.5 + 0.5
    res = tensor.clone()
    res = res.reshape(-1,3,128,128)
    res = torch.sum(res,1)
    res = res.reshape(-1,1,128,128)
    res=  torch.cat((res,res,res),dim=1)
    img_binary = custom_replace(res,0,1)
    return img_binary


# def save_from_mask(dataloader,save_path):
#     if os.path.isdir(save_path)==False:
#         os.mkdir(save_path)
#     for data in dataloader:
#         img, img_name, img_box, _2 = data # use original images
#         img = img
#         img = img.view(img.size(0), -1)
#         if norm == 'Y':
#             img = img*0.5+0.5
#         img_binary = custom_replace_sumdim(img,0,1)
#         img_binary = img_binary.reshape(-1,3,128,128)
#         # now save them all
#         for i, i_name in zip(img_binary,img_name):
#             #i = i*0.5+0.5
#             save_image(i, save_path+i_name)
#             #save_image(i, project_path+'models/'+test_filename+'/output_imgs/'+i_name)
#         del img, img_name, img_box, _2, img_binary
#         print('dl complete')
#
#
# save_from_mask(train_set_loader, project_path+'/custom_masks/')
# save_from_mask(test_set_loader, project_path+'/custom_masks/')
# save_from_mask(tex_set_loader, project_path+'/tex_masks/')
#

############################
# visualisation function
# original
############################


def show_image(img):
    img = img.view(3,128,128)
    if float(img.min())<0:
        img = img * 0.5 + 0.5
    # Convert to numpy for visualisation
    try:
        npimg = img.numpy()
    except:
        # if variable
        npimg= img.detach().numpy()
    # Plot each image
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



#****************************************************************************
# MODELS
#****************************************************************************

class autoencoder_PCA(nn.Module):

    def __init__(self,num_in,num_hidden_1,num_out):
        super(autoencoder_PCA, self).__init__()

        # The layers of the encoder
        self.encoder = nn.Linear(num_in, num_hidden_1)

        # The layers of the decoder
        self.decoder = nn.Sequential(
            nn.Linear(num_hidden_1, num_out),
            nn.Tanh())

    def forward(self, x):
        ''' try the binary after training'''
        x_unorm = x.data*0.5+0.5
        img_binary = custom_replace(x_unorm,0,1)
        orig_x = x.data*img_binary
        x = self.encoder(x)
        x = self.decoder(x)
        loss_x = x.data*img_binary
        loss_x = Variable(loss_x)
        orig_x = Variable(orig_x, requires_grad=True)
        return x, loss_x, orig_x

    def forward(self, x):
        ''' binary before the training'''
        x_unorm = x.data*0.5+0.5
        img_binary = custom_replace(x_unorm,0,1)
        x = x.data*img_binary
        x = Variable(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward(self, x):
        ''' Original '''
        x = self.encoder(x)
        x = self.decoder(x)
        return x




class deep_autoencoder(nn.Module):
    def __init__(self,num_in,num_hidden_1,num_hidden_2,num_hidden_3,num_out):
        super(deep_autoencoder, self).__init__()

        # The layers of the encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_in, num_hidden_1),
            nn.ReLU(True),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.ReLU(True),
            nn.Linear(num_hidden_2, num_hidden_3))

        # The layers of the dencoder
        self.decoder = nn.Sequential(
            nn.Linear(num_hidden_3, num_hidden_2),
            nn.ReLU(True),
            nn.Linear(num_hidden_2, num_hidden_1),
            nn.ReLU(True),
            nn.Linear(num_hidden_1, num_out),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNN_autoencoder(nn.Module):
    def __init__(self,num_in,num_hidden_1,num_hidden_2,num_hidden_3,num_out):
        super(CNN_autoencoder, self).__init__()

        # The layers of the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(num_in, num_hidden_1, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_hidden_1, num_hidden_2, 2, 2, 0),
            nn.BatchNorm2d(num_hidden_2),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),
            nn.Conv2d(num_hidden_2, num_hidden_3, 3, 1, 1),
            nn.BatchNorm2d(num_hidden_3),
            nn.LeakyReLU(0.2, inplace=True))

        # self.e_1 = nn.Conv2d(num_in, num_hidden_1, 3, 1, 1)
        # self.e_2 = nn.Conv2d(num_hidden_1, num_hidden_2, 2, 2, 0)
        # self.e_3 = nn.Conv2d(num_hidden_2, num_hidden_3, 3, 1, 1)


        # self.d_1 = nn.Conv2d(num_hidden_3, num_hidden_2, 3, 1, 1)
        # self.d_2 = nn.ConvTranspose2d(num_hidden_2, num_hidden_1, 2, 2, 0)
        # self.d_3 = nn.Conv2d(num_hidden_1, num_out, 3, 1, 1)

        # The layers of the dencoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_hidden_3, num_hidden_2, 3, 1, 1),
            nn.BatchNorm2d(num_hidden_2),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hidden_2, num_hidden_1, 2, 2, 0),
            nn.ReLU(True),
            nn.BatchNorm2d(num_hidden_1),
            nn.ConvTranspose2d(num_hidden_1, num_out, 3, 1, 1),
            nn.Tanh())

    def forward(self, x):

        # x = self.e_1(x)
        # #print("e1:", x.shape)
        # x = self.e_2(x)
        # #print("e2:", x.shape)
        # x = self.e_3(x)
        # #print("e3:", x.shape)
        #
        # x = self.d_1(x)
        # #print("d1:", x.shape)
        # x = self.d_2(x)
        # #print("d2:", x.shape)
        # x = self.d_3(x)
        # #print("d3:", x.shape)

        x = self.encoder(x)
        x = self.decoder(x)

        return x



#****************************************************************************
# LOSS FUNCTIONS
#****************************************************************************

# criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# criterion = nn.BCELoss()

def mse_loss(input, target):
    """ Not working """
    loss_ =  torch.sum((input - target).data * (input - target).data)/input.data.nelement()
    loss_ = numpy.array([loss_])
    loss_ = torch.autograd.Variable(torch.from_numpy(loss_))
    return loss_

def weighted_mse_loss(input, target, weights):
    input = input.cuda()
    target = target.cuda()
    weights =  weights.cuda()
    # out = input - target
    out = (input-target)**2
    out = out * weights
    #loss = out.sum(0) # or sum over whatever dimensions
    return out.sum()/input.data.nelement()

def norm_mse_loss(input, target, weights):
    global s, w_s, weights_1, out, input_, target_
    input_=input
    target_ = target
    input = input.reshape(input.shape[0],3,128,128)
    # print("input:",input.shape)
    target = target.reshape(input.shape[0],3,128,128)
    # print("target:", target.shape)
    weights_1 = weights.reshape(input.shape[0],3,128,128)
    # print("weights:", target.shape)

    out = (input-target)**2

    # img_t = img_t*0.5+0.5

    s_1 = torch.sum(weights_1,1)
    s_1_n = np.asarray(s_1)

    count = np.count_nonzero(s_1_n, axis = (1,2))
    s = 1/(count/(128*128))

    s = torch.Tensor(s)
    # print('s:', s.shape)
    # weights = weights.reshape(weights.shape[0], -1)
    # print('weights_orig:', weights_1[0])
    # s=s.reshape(s.shape,1)
    s = s.view(s.shape[0],1,1,1)
    # print(s[0])
    w_s = weights_1*s
    # print('weights_new:',w_s[0])
    out_f = out * w_s

    out_f = out_f.sum()

    return out_f/input.data.nelement()

# class weighted_mse_loss(nn.Module):
#     def __init__(self, extractor):
#         super(weighted_mse_loss, self).__init__()
#         self.l1 = nn.L1Loss()
#         self.extractor = extractor
#     def forward(self, input, target, weights):
#         out = (input-target)**2
#         out = out * weights
#         return out.sum()/input.data.nelement()



def customBCELoss(outputs, truths):
    """ Returns inf  """
    loss_BCE = (truths * torch.log(outputs) + torch.add(torch.neg(truths), 1.) * torch.log(torch.add(torch.neg(outputs), 1.)) )
    avg_BCE = torch.neg(torch.mean(loss_BCE))
    return avg_BCE


def mse_identity(input, target, weights, weights_i):
    out = (input-target)**2
    loss_1 = out * weights
    loss_1 = loss_1.sum()/input.data.nelement()
    loss_2 = out * weights_i.expand_as(out)
    loss_2 = loss_2.sum()/input.data.nelement()
    #loss = out.sum(0) # or sum over whatever dimensions
    loss = (0.6*loss_1) + (0.4*loss_2)
    return loss



#****************************************************************************
# TRAIN FUNCTIONS
#****************************************************************************

def train(train_set, batch_size, num_epochs, learning_rate, model):
    """ Train the encoder and return the losses"""
    no_batches = int(np.round((len(train_set) / batch_size)))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

    losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        for data in train_set_loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img)

            # =================== forward pass =====================
            output = model(img)
            loss = criterion(output, img)

            # =================== backward pass ====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]

            # =================== log ========================
        print('====> Epoch: {} Average loss: {:.4f}'.format(
               epoch, train_loss / no_batches))
        losses.append(train_loss / no_batches)
    # add save loss to csv
    return losses



def evaluate_test(test_set, test_set_loader, batch_size, model, loss_function, box = False, flatten=True):
    test_loss = 0
    # criterion = nn.MSELoss()
    test_no_batches = np.round(int(len(test_set)/batch_size))
    # print("test_no_batches:", test_no_batches)
    losses_test = []
    #loss_function = loss_function.cuda()
    model.eval()
    for data in test_set_loader:
        if box == False:
            img, _, _1, img_binary = data # use original images
            if norm == 'Y':
                img = img*0.5+0.5
            if flatten==True:
                img = img.view(img.size(0), -1)
                img_binary = img_binary.view(img_binary.size(0), -1)
        elif box == True:
            orig_img, _, img, img_binary = data # use box images
            if norm == 'Y':
                img = img*0.5+0.5
            if flatten==True:
                orig_img = orig_img.view(orig_img.size(0), -1)
                img = img.view(img.size(0), -1)
                img_binary = img_binary.view(img_binary.size(0), -1)
            orig_img = Variable(orig_img)
            #img_binary = img_binary*0.5+0.5
        img_binary=img_binary.cuda()
        img = Variable(img)
        img = img.cuda()
        img_binary = img_binary.cuda()

        # =================== forward pass =====================
        output = model(img)
        if box == False:
            loss = loss_function(output, img, Variable(img_binary))
        elif box == True:
            loss = loss_function(output, orig_img, Variable(img_binary))
        loss = loss.detach()
        test_loss += float(loss.data[0])
        # =================== log ========================
        # del img, _, _1, img_binary, loss

    test_loss = float(test_loss)/float(test_no_batches)
    test_loss=float(test_loss)
    imgs_to_save = torch.cat((img,output,img_binary), dim=0)
    del img, _, _1, img_binary, loss
    return test_loss, imgs_to_save



def save_from_dataloader(dataloader,save_path,model):
    print('started load')
    if os.path.isdir(save_path)==False:
        os.mkdir(save_path)
    model.eval()
    counter=0
    for data in dataloader:
        counter+=1
        img, img_name, img_box, _2 = data # use original images
        img = img.cuda()
        img = img.view(img.size(0), -1)
        # if norm == 'Y':
        #     img = img*0.5+0.5
        output = model(img)
        output = output.reshape(-1,3,128,128)
        # now save them all

        for i, i_name in zip(output,img_name):
            i = i*0.5+0.5
            save_image(i, save_path+i_name)
            #save_image(i, project_path+'models/'+test_filename+'/output_imgs/'+i_name)
        del img, img_name, img_box, _2, output
        # print('Saved Batch:',counter)
#print('SAVED PATH', save_path)


# Final well defined train
def train_nonzero(train_set,
                  model,
                  batch_size=128,
                  num_epochs=1000,
                  learning_rate=1e-2,
                  print_examples="Y",
                  test_filename='file_name_here',
                  box = False,
                  flatten=True,
                  loss_function=nn.MSELoss()
                  ):
    #loss_function = loss_function.cuda()
    # Create save folder
    if os.path.exists(project_path+'models/'+test_filename)==False:
        os.mkdir(project_path+'models/'+test_filename)

    model = model.to(device)
    criterion = loss_function

    no_batches = int(np.round((len(train_set) / batch_size)))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(0, args.max_iter)):
        print('epoch',epoch)
        train_loss = 0
        for data in train_set_loader:
            model.train()
            if box == False:
                img, _, _1, img_binary = data
                img = img.cuda()
                img_binary = img_binary.cuda()

                if flatten==True:
                    img = img.view(img.size(0), -1)
                    img_binary = img_binary.view(img_binary.size(0), -1)
                if norm == 'Y':
                    img = img*0.5+0.5

            # elif box == True:
            #     orig_img, _, img, img_binary = data # use box images
            #     orig_img = orig_img.cuda()
            #     img = img.cuda()
            #     img_binary = img_binary.cuda()
            #     if flatten==True:
            #         img = img.view(img.size(0), -1)
            #         orig_img = orig_img.view(img.size(0), -1)
            #         img_binary = img_binary.view(img_binary.size(0), -1)
            #         if norm == 'Y':
            #             img = img*0.5+0.5

            # =================== forward pass =====================
            # img = Variable(img)
            output = model(img)

            # BESPOKE LOSS FUNCTION: Feed in non zero weights
            if box == False:
                loss = criterion(output, img, Variable(img_binary))
            elif box == True:
                # orig_img = Variable(orig_img)
                loss = criterion(output, orig_img, Variable(img_binary))

            # =================== backward pass ====================

            train_loss += float(loss.data)
            # print('loss:',float(loss.data))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            del img,output,loss

        model.eval()
        test_loss, imgs_to_save = evaluate_test(test_set, test_set_loader, len(test_set), model, loss_function, box = box, flatten=flatten)

        # # save best model
        # if epoch ==0:
        #     #torch.save(model.state_dict(), project_path+'models/'+test_filename+'/best_model.pth')
        #     # model.save_state_dict(project_path+'models/'+test_filename+'/best_model.pt')
        #     pass
        # elif test_loss < min(test_losses):
        #     print("++++++++++++++++++++ MODEL IMPROVED +++++++++++++++++++++++")
        #     torch.save(model.state_dict(), project_path+'models/'+test_filename+'/best_model.pth')
        #     # model.save_state_dict(project_path+'models/'+test_filename+'/best_model.pt')
        # else:
        #     print('-- NO IMPROVEMENT')
        #
        # print('Model Saved')
        print('====> Epoch: {} Train loss: {:.4f} Test loss: {:.4f}'.format(
              epoch, train_loss / no_batches, test_loss))
        train_losses.append(train_loss / no_batches)
        test_losses.append(test_loss)

        # # Actual vs Models Example Images
        # try:
        #     plt.close(fig)
        # except:
        #     pass
        # fig = plt.figure(figsize=(10,5))
        # train_img2 = img[-1]*0.5+0.5
        # train_img2=train_img2.data.numpy()
        # train_img2 = train_img2.reshape(3,128,128)
        # train_img2 = np.transpose(train_img2, (1, 2, 0))
        # ax1 = fig.add_subplot(2,2,1)
        # ax1.imshow(train_img2)
        #
        # # Show the modelled
        # model_img2 = output[-1]*0.5+0.5
        # model_img2 = model_img2.data.numpy()
        # model_img2 = model_img2.reshape(3,128,128)
        # model_img2 = np.transpose(model_img2, (1, 2, 0))
        # ax2 = fig.add_subplot(2,2,2)
        # ax2.imshow(model_img2)
        # if print_examples =='Y':
        #     plt.show()
        # fig.savefig(project_path+'models/'+test_filename+'/epoch'+str(epoch)+'.png', bbox_inches='tight')

        #Plot and save losses
        a_plot = two_plots(train_losses, 'train_losses', test_losses , 'test_losses', 'epoch',
                           title = 'Test loss vs Train loss',
                           save_name=project_path+'models/'+test_filename+'/loss_plt.png',
                           show_plot=False)
        # try:
        #
        #     plt.close()
        # except:
        #     pass

        # save losses to csv
        loss_df = pd.DataFrame(train_losses, test_losses)
        loss_df = loss_df.reset_index()
        loss_df.columns = ['test_loss','train_loss']
        loss_df.to_csv(project_path+'models/'+test_filename+'.csv')

        num = epoch+1
        if num%args.vis_interval==0:
            print('Saving Epoch',epoch,project_path+'models/'+test_filename+'/output_imgs_'+str(num)+'/' )
            save_path2 = project_path+'models/'+test_filename+'/output_imgs_'+str(num)+'/'
            print('path define')

            save_image(imgs_to_save, project_path+'models/'+test_filename+'/loss'+str(num)+'_plt.png')
            print('image saved')
            a_plot = two_plots(train_losses, 'train_losses', test_losses , 'test_losses', 'epoch',
                               title = 'Test loss vs Train loss',
                               save_name=project_path+'models/'+test_filename+'/loss'+str(num)+'_plt.png',
                               show_plot=False)
            # print('plot made')
        if num%args.save_images_interval==0:
            # print('running save_from_dataloader')
            save_from_dataloader(test_set_loader,save_path2,model)
            print('TEST SET SAVED')
            save_from_dataloader(train_set_loader,save_path2,model)
            print('TRAIN SET SAVED')
            save_from_dataloader(tex_set_loader,save_path2+'/tex_run/',model)
            print('TEX SET SAVED')
            save_ckpt(project_path+'models/'+test_filename+'/'+str(num)+'.pth',[('model', model)], [('optimizer', optimizer)], num)
    return train_losses, test_losses, model




# #****************************************************************************
# # TRAIN MODELS
# #****************************************************************************
#
# # Run the PCA loop
# num_input=3*128*128
# num_output=3*128*128
# test_layers = [32,64,128,256]
# for l in test_layers:
#     model1 = autoencoder_PCA(num_input,l,num_output)
#     import time
#     start = time.time()
#     print("*"*200)
#     print('running size',l)
#     train_losses_rtn, test_losses_rtn, model_rtn = train_nonzero(train_set,
#                                                                   model1,
#                                                                   batch_size=128,
#                                                                   num_epochs=2001,
#                                                                   # learning_rate=1e-3,
#                                                                   learning_rate=2e-4,
#                                                                   print_examples="N",
#                                                                   test_filename='RUN_PCA_full_'+str(l),
#                                                                   box = False,
#                                                                   flatten=True,
#                                                                   loss_function=weighted_mse_loss
#                                                                   )
#     end = time.time()
#     hours, rem = divmod(end-start, 3600)
#     minutes, seconds = divmod(rem, 60)
#     #print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
#     print('Complete in '+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
#
#
#



##############################
# PCA: Single
##############################
num_input=3*128*128
num_hidden_1=args.num_hidden_neurons
num_output=3*128*128
model1 = autoencoder_PCA(num_input,num_hidden_1,num_output)
train_nonzero(train_set,
                                                              model1,
                                                              batch_size=args.batch_size,
                                                              num_epochs=args.max_iter,
                                                              # learning_rate=1e-3,
                                                              learning_rate=2e-4,
                                                              print_examples="N",
                                                              test_filename='RUN_19K_PCA_full_'+str(num_hidden_1),
                                                              box = False,
                                                              flatten=True,
                                                              loss_function=weighted_mse_loss
                                                              )


print('COMPLETE')



#
# ##############################
# # DEEP LEARNING: Single
# ##############################
# l = [[64,32,16,'deep64_32_16']
# model = deep_autoencoder(3*128*128,l[0],l[1],l[2],3*128*128)
# train_losses_rtn, test_losses_rtn, model_rtn = train_nonzero(train_set,
#                                                               model1,
#                                                               batch_size=128,
#                                                               num_epochs=1000,
#                                                               learning_rate=1e-3,
#                                                               print_examples="N",
#                                                               test_filename='orig_mse_full_mask_deep64_32_16',
#                                                               box = False,
#                                                               flatten=True,
#                                                               loss_function=weighted_mse_loss
#                                                               )
#
#
#
# ##############################
# # DEEP LEARNING: Loops
# ##############################
# test_layers = [[64,32,16,'deep64_32_16'],[128,64,32,'deep128_64_32'],[256,128,64,'deep_256_128_64'],[256,128,32,'deep_256_128_32']]
#
# # Run the above
# for l in test_layers:
#     import time
#     start = time.time()
#     print("*"*200)
#     print('running',l[3])
#     model = deep_autoencoder(3*128*128,l[0],l[1],l[2],3*128*128)
#     train_losses_rtn, test_losses_rtn, model_rtn = train_nonzero(train_set,
#                                                                   model,
#                                                                   batch_size=128,
#                                                                   num_epochs=1000,
#                                                                   learning_rate=1e-3,
#                                                                   print_examples="N",
#                                                                   test_filename='ann_orig_mse_full_mask_'+str(l[3]),
#                                                                   box = False,
#                                                                   flatten=True,
#                                                                   loss_function=weighted_mse_loss
#                                                                   )
#     end = time.time()
#     hours, rem = divmod(end-start, 3600)
#     minutes, seconds = divmod(rem, 60)
#     #print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
#     print('Complete in '+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
#
#
#
#
# ##############################
# # CNN: Single
# ##############################
# model = CNN_autoencoder(3,3,32,32,3)
# model = CNN_autoencoder(3,3,64,64,3)
# train_losses_rtn, test_losses_rtn, model_rtn = train_nonzero(train_set,
#                                                               model,
#                                                               batch_size=128,
#                                                               num_epochs=1000,
#                                                               learning_rate=1e-3,
#                                                               print_examples="N",
#                                                               test_filename='orig_mse_full_mask_cnn32_3',
#                                                               box = True,
#                                                               flatten=False,
#                                                               loss_function=weighted_mse_loss
#                                                               )
#
#
