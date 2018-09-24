from __future__ import print_function

# from tqdm import tqdm
import torch
import torch as nn
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import os
import copy
import project_data
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.utils import save_image
import pandas as pd
# reload(project_data)
from torch import nn


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




cwd = os.getcwd()
if 'ubuntu' in cwd:
    project_path = os.getcwd()+'/'
    data_path = '/home/ubuntu/data/'
    print('ProjectData project_path = ',cwd)
else:
    project_path = 'path'



parser = argparse.ArgumentParser()
# training options
# parser.add_argument('--NOTES', type=str,default='no notes')
# parser.add_argument('--root', type=str, default='/srv/datasets/Places2')
# parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
# parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--lr_finetune', type=float, default=1e-4)
parser.add_argument('--max_iter', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=100)
parser.add_argument('--vis_interval', type=int, default=100)
# parser.add_argument('--log_interval', type=int, default=1000)
# parser.add_argument('--image_size', type=int, default=128)
# parser.add_argument('--resume', type=str)
# parser.add_argument('--finetune', action='store_true')
parser.add_argument('--gpu', type=str,default=True)
args = parser.parse_args()

print('*'*50)
print('--- ARGS ---')
print('lr',args.lr)
print('batch_size',args.batch_size)
print('gpu',args.gpu)
print('gpu',type(args.gpu))
print('save_model_interval',(args.save_model_interval))
print('vis_interval',(args.vis_interval))
print('*'*50)


class cnn_blend(nn.Module):
    def __init__(self):
        super(cnn_blend, self).__init__()

        self.channels = 3
        self.layer1 = nn.Sequential(
                    nn.Conv2d(self.channels, 16, 3, stride=1, padding=1),
                    nn.LeakyReLU(0.2))


        self.layer2 = nn.Sequential(
                    nn.Conv2d(16, 3, 3, stride=1, padding=1),
                    nn.BatchNorm2d(3, 0.8),nn.LeakyReLU(0.2),
                    nn.LeakyReLU(0.2))
        # self.encoder = nn.Linear(128, 128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, to_concat):
        residual = to_concat

        x = self.layer1(x)
        # print("e_1:" ,x.shape)
        x = self.layer2(x)
        # print("e_2:" ,x.shape)

        # x = self.encoder(x)

        x += residual
        x = self.relu(x)
        return x

#Alternative Approach
# class cnn_blend(nn.Module):
#     def __init__(self):
#         super(cnn_blend, self).__init__()

#         self.channels = 3
#         self.layer1 = nn.Sequential(
#                     nn.Conv2d(self.channels, 16, 3, stride=1, padding=1),
#                     nn.LeakyReLU(0.2))


#         self.layer2 = nn.Sequential(
#                     nn.Conv2d(16, 3, 3, stride=1, padding=1),
#                     nn.BatchNorm2d(3, 0.8),nn.LeakyReLU(0.2),
#                     nn.LeakyReLU(0.2))
#         # self.encoder = nn.Linear(128, 128)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, conv, pca):

#         inp_1 = conv + pca

#         # residual = to_concat

#         out_1 = self.layer1(inp_1)

#         inp_2 = out_1 + inp_1
#         out_2 = self.layer2(inp_2)

#         inp_3 = out_2 + inp_2
#         x = self.relu(inp_3)

#         return x

cnn = cnn_blend()

dataset_train, dataset_val = project_data.data_gen('Y')
iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    num_workers=args.n_threads))

def custom_replace(tensor, on_zero, on_non_zero):
    """ Create Tensor with binary values
        Rule: Non Zero
    """
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res


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


train_set_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=1,
                                               shuffle=True)

test_set_loader = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=1,
                                              shuffle=True)


def weighted_mse_loss(input, target, weights):
    input = input.cuda()
    target = target.cuda()
    weights =  weights.cuda()
    # out = input - target
    out = (input-target)**2
    out = out * weights
    #loss = out.sum(0) # or sum over whatever dimensions
    return out.sum()/input.data.nelement()



def evaluate(model, img_name):
    test_set_loader = torch.utils.data.DataLoader(dataset_val,
                                                  # batch_size=len(dataset),
                                                  batch_size=1,
                                                  shuffle=True)
    data_iterator_test = iter(test_set_loader)

    counter=0
    test_loss = 0
    model.eval()
    for data in data_iterator_test:
        counter+=1
        if counter <9:
            gt, label, pca_img, pconv_img = data
            # gt, _, pca_img, pconv_img
            gt = gt.cuda()
            pca_img=pca_img.cuda()
            pconv_img=pconv_img.cuda()
            gt_mask = custom_replace(gt, 0, 1)

            with torch.no_grad():
                output = cnn(pconv_img, pca_img)
                loss = weighted_mse_loss(output, gt, Variable(gt_mask))
                test_loss += loss.data[0]

            if counter>1:
                # print('>1 -----------')
                gt_concat = torch.cat((gt_concat, gt.data))
                output_concat = torch.cat((output_concat, output.data))
                pca_img_concat = torch.cat((pca_img_concat, pca_img.data))
                pconv_img_concat = torch.cat((pconv_img_concat, pconv_img.data))
                label_concat = label_concat +[label[0]]
            else:
                # print('f ----------------')
                gt_concat = gt.data
                pca_img_concat = pca_img.data
                pconv_img_concat = pconv_img.data
                # image_concat = image.data
                output_concat = output.data
                label_concat = [label[0]]
            del gt, pca_img, pconv_img, output, loss


    test_loss = test_loss/16
    test_loss=float(test_loss)

    grid = make_grid(
    torch.cat((gt_concat, pca_img_concat,pconv_img_concat,output_concat ), dim=0)) #image_concat  --do we need this?
    save_image(grid, project_path+''+img_name)
    print('EVAL COMPLETE')

    # Epochs are finished
    if os.path.isdir(project_path+'models/output_imgs/')==False:
        os.mkdir(project_path+'models/output_imgs/')

    del gt_concat, pca_img_concat,pconv_img_concat,output_concat
    return test_loss



#### CHANGE THIS NOW!
device = torch.device('cuda')
# set up dirs
if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))



def run_style_transfer(model, test_filename='file_name_here',learning_rate = 1e-3, num_epochs=1000, batch_size=128):
        """Run the style transfer."""
        print('Building the style transfer model..')
        if os.path.exists(project_path+'models/')==False:
            os.mkdir(project_path+'models/')
        if os.path.exists(project_path+'models/'+test_filename)==False:
            os.mkdir(project_path+'models/'+test_filename)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        train_losses = []
        test_losses = []
        no_batches = int(np.round((len(dataset_train) / batch_size)))

        for epoch in range(num_epochs):
            model = model.to(device)
            print(epoch)
            train_loss = 0
            for data in train_set_loader:
                # print('len loader',len(data))
                gt, _, pca_img, pconv_img = data
                gt_mask = custom_replace(gt, 0, 1)
                pca_mask = custom_replace(pca_img, 0, 1)
                pconv_mask = custom_replace(pconv_img, 0, 1)

                gt = gt.cuda()
                pca_img = pca_img.cuda()
                pconv_img = pconv_img.cuda()
                pconv_mask = pconv_mask.cuda()
                gt_mask = gt_mask.cuda()
                pca_mask = pca_mask.cuda()


                # pred_concat = pca_img+gt

                output = model(pconv_img, pca_img)

                loss = weighted_mse_loss(output, gt, Variable(gt_mask))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.data[0]
                train_loss = float(train_loss)
            save_img_name = 'test_'+str(epoch)+'.jpg'

            test_loss = evaluate(model,save_img_name)
            print('====> Epoch: {} Train loss: {:.4f} Test loss: {:.4f}'.format(epoch, train_loss / no_batches, test_loss))
            train_losses.append(train_loss / no_batches)
            test_losses.append(test_loss)
            a_plot = two_plots(train_losses, 'train_losses', test_losses , 'test_losses', 'epoch',
                               title = 'Test loss vs Train loss',
                               save_name=project_path+'models/'+test_filename+'/loss_plt.png',
                               show_plot=False)

            loss_df = pd.DataFrame(train_losses, test_losses)
            loss_df = loss_df.reset_index()
            loss_df.columns = ['test_loss','train_loss']
            loss_df.to_csv(project_path+'models/'+test_filename+'.csv')

            num = epoch+1
            if num%args.vis_interval==0:
                print('Saving Epoch',epoch,project_path+'models/'+test_filename+'/output_imgs_'+str(num)+'/' )
                save_path2 = project_path+'models/'+test_filename+'/output_imgs_'+str(num)+'/'
                print('path define')

                # save_image(imgs_to_save, project_path+'models/'+test_filename+'/loss'+str(num)+'_plt.png')
                print('image saved')
                a_plot = two_plots(train_losses, 'train_losses', test_losses , 'test_losses', 'epoch',
                                   title = 'Test loss vs Train loss',
                                   save_name=project_path+'models/'+test_filename+'/loss'+str(num)+'_plt.png',
                                   show_plot=False)

            if num%args.save_model_interval==0:
                save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, num),
                          [('model', model)], [('optimizer', optimizer)], num)


train_losses, test_losses, model = run_style_transfer(cnn, "PCA-PCONV-BLENDED", learning_rate=args.lr, num_epochs=args.max_iter, batch_size =16)
