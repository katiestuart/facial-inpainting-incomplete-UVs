import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from torchvision.utils import save_image
import sys
sys.path.append("/home/ubuntu/data/")
sys.path.append('path')
import opt
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor


cwd = os.getcwd()
if 'ubuntu' in cwd:
    project_path = os.getcwd()+'/'
    data_path = '/home/ubuntu/data/'
    print('ProjectData project_path = ',cwd)
else:
    project_path = 'path'

sys.path.append(cwd)


import project_data

def weighted_mse_loss(input, target, weights):
    out = (input-target)**2
    out = out * weights
    return out.sum()/input.data.nelement()

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


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


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x.cpu() * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
    x = x.transpose(1, 3)
    return x


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--NOTES', type=str,default='no notes')
parser.add_argument('--root', type=str, default='/srv/datasets/Places2')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_finetune', type=float, default=1e-4)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=1000)
parser.add_argument('--vis_interval', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=1000)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--gpu', type=str,default=False)
args = parser.parse_args()


print('*'*50)
print('--- ARGS ---')
print('lr',args.lr)
print('lr_finetune',args.lr_finetune)
print('batch_size',args.batch_size)
print('gpu',args.gpu)
print('gpu',type(args.gpu))
print('*'*50)


# class Namespace:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
# args = Namespace(a=1, b='c')
# args.a

# args =  Namespace(
#             mask_root='./masks'
#             max_iter=50,
#             root = '/srv/datasets/Places2',
#             lr = 2e-4,
#             lr_finetune = 5e-5,
#             batch_size = 16,
#             n_threads = 16,
#             save_model_interval = 50000,
#             vis_interval = 1,
#             log_interval = 10,
#             image_size = 128,
#             finetune = 'store_true',
#             save_dir = './snapshots/default',
#             log_dir = './logs/default',
#             resume = str,
#             start_iter = 0,
#             args.gpu = False)

# args.max_iter



import json
args_dict = vars(args)
with open('file.txt', 'w') as file:
     file.write(json.dumps(args_dict))

fout = "model_parameters.txt"
fo = open("model_parameters.txt", "w")
for k, v in args_dict.items():
    fo.write(str(k) + ': '+ str(v) + '\n')
fo.close()



##############
# GPU
##############
if args.gpu=='True':
    device = torch.device('cuda')
    print('GPU DEVICE')


# set up dirs
if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
try:
    os.mkdir(project_path+'models/')
except:
    pass

#################
# Load in data
#################
writer = SummaryWriter(log_dir=args.log_dir)
size = (args.image_size, args.image_size)
img_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])


dataset_train, dataset_val = project_data.data_gen('Y')
iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads))

iterator_test = iter(data.DataLoader(
    dataset_val, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_val)),
    num_workers=args.n_threads))

#################
# Define model
#################
if args.gpu=='True':
    model = PConvUNet().to(device)
else:
    model = PConvUNet()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)


if args.gpu=='True':
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)
else:
    criterion = InpaintingLoss(VGG16FeatureExtractor())


if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

# resume training from last time
if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('optimizer', optimizer)])
else:
    start_iter = 0


#################
# train & eval
#################
test_loss=[]
train_loss=[]
for i in tqdm(range(start_iter, args.max_iter)):
    model.train()
    gt, label, image, mask = [x for x in next(iterator_train)]
    gt = gt.cuda()
    image=image.cuda()
    mask=mask.cuda()

    # if args.gpu=='True':
    #     gt, _, image, mask = [x.to(device) for x in next(iterator_train)]
    # else:
    #     gt, _, image, mask = [x for x in next(iterator_train)]

    output, _ = model(image, mask)
    loss_dict = criterion(image, mask, output, gt)

    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
    #     save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
    #               [('model', model)], [('optimizer', optimizer)], i + 1)

    if (i + 1) % args.vis_interval == 0:
        model.eval()
        if args.gpu =='True':
            # print('GPU iterator_train')
            loss_t = evaluate(model, dataset_val,
                 '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1),GPU_YN_IN=True)
        else:
            print('GPU False')
            loss_t = evaluate(model, dataset_val,
                 '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1),GPU_YN_IN=False)


    # Store loss
    loss_int = float(loss)
    train_loss.append(loss_int)
    loss_df = pd.DataFrame(train_loss)
    loss_df = loss_df.reset_index()
    loss_df.to_csv('train_loss.csv')

    if (i + 1) % args.vis_interval == 0:
        loss_t = float(loss_t)
        test_loss.append(loss_t)
        test_loss_df = pd.DataFrame(test_loss)
        test_loss_df = test_loss_df.reset_index()
        test_loss_df.to_csv('test_loss.csv')
    del output, _, gt, image, mask

# Epochs are finished
if os.path.isdir(project_path+'models/output_imgs/')==False:
    os.mkdir(project_path+'models/output_imgs/')
    os.mkdir(project_path+'models/output_imgs/test/')
    os.mkdir(project_path+'models/output_imgs/train/')
    os.mkdir(project_path+'models/output_imgs/train_mask/')

# gt, _, image, mask, pca = [x.to(device) for x in next(iterator_train)]

#Store final images
norm = 'Y'
# output = output.reshape(-1,3,128,128)
# for i, i_name in zip(output,label):
#     save_image(i, project_path+'models/output_imgs/test/'+i_name)


iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=1,shuffle=True))

iterator_test = iter(data.DataLoader(
    dataset_val, batch_size=1,shuffle=True))

print(args.batch_size)

print('Now Saving final modelled images ...')
model.eval()
counter=0
for data in iterator_train:
    counter+=1
    model.eval()
    print(counter)
    gt, label, img, mask  = data # use original images
    img = img.cuda()
    gt = gt.cuda()
    mask = mask.cuda()

    output, _ = model(img, mask)
    # img1 = img
    img = img.view(img.size(0), -1)
    if norm == 'Y':
        img = img*0.5+0.5

    output = output.reshape(-1,3,128,128)
    # now save them all
    for output_i, output_name in zip(output,label):
        #counter+=1
        save_image(output_i, project_path+'models/output_imgs/'+output_name)

    del img,output,_,gt
print('Train Images Saved: ',print(counter))


print('Now Saving final TEST modelled images ...')
model.eval()
counter=0
for data in iterator_test:
    counter+=1
    model.eval()
    print(counter)
    gt, label, img, mask  = data # use original images
    img = img.cuda()
    gt = gt.cuda()
    mask = mask.cuda()

    output, _ = model(img, mask)
    # img1 = img
    img = img.view(img.size(0), -1)
    if norm == 'Y':
        img = img*0.5+0.5

    output = output.reshape(-1,3,128,128)
    # now save them all
    for output_i, output_name in zip(output,label):
        #counter+=1
        save_image(output_i, project_path+'models/output_imgs/'+output_name)

    del img,output,_,gt
print('TEST Images Saved:',print(counter))


save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)


# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
# from torch.autograd import Variable
#
# for i in tqdm(range(start_iter, max_iter)):
#     model.train()
#     #image, mask, gt = [x.to(device) for x in next(iterator_train)]
#     gt, _, image, mask = [x for x in next(iterator_train)]
#     gt = gt*0.5+0.5
#     gt_v = Variable(gt)
#     gt_v = gt_v.data.numpy()
#     gt_v = gt_v[0]
#     gt_v = gt_v.reshape(3,128,128)
#     gt_v = np.transpose(gt_v, (1, 2, 0))
#     plt.imshow(gt_v)
#     plt.show()
#
#     image = image*0.5+0.5
#     image_v = Variable(image)
#     image_v = image_v.data.numpy()
#     image_v = image_v[0]
#     image_v = image_v.reshape(3,128,128)
#     image_v = np.transpose(image_v, (1, 2, 0))
#     plt.imshow(image_v)
#     plt.show()
    #
    # mask = mask*0.5+0.5
    # mask_v = Variable(mask)
    # mask_v = mask_v.data.numpy()
    # mask_v = mask_v[0]
    # mask_v = mask_v.reshape(3,128,128)
    # mask_v = np.transpose(mask_v, (1, 2, 0))
    # plt.imshow(mask_v)
    # plt.show()
    #
    #
    # output, _ = model(image, mask)
    # #mask = mask*0.5+0.5
    # output_v = Variable(output)
    # output_v = output_v.data.numpy()
    # output_v = output_v[0]
    # output_v = output_v.reshape(3,128,128)
    # output_v = np.transpose(output_v, (1, 2, 0))
    # plt.imshow(output_v)
    # plt.show()
    #
    # loss_dict = criterion(image, mask, output, gt)
    #
    # loss = 0.0
    # for key, coef in opt.LAMBDA_DICT.items():
    #     value = coef * loss_dict[key]
    #     loss += value
    #     if (i + 1) % log_interval == 0:
    #         writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)
    #
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    #
    # if (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
    #     save_ckpt('{:s}/ckpt/{:d}.pth'.format(save_dir, i + 1),
    #               [('model', model)], [('optimizer', optimizer)], i + 1)
    #
    # if (i + 1) % vis_interval == 0:
    #     model.eval()
    #     evaluate(model, dataset_val,
    #              '{:s}/images/test_{:d}.jpg'.format(save_dir, i + 1))

# writer.close()
