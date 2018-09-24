import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
from loss import InpaintingLoss
from net import VGG16FeatureExtractor
import opt
import os

#add util
import sys
sys.path.append('path')
sys.path.append("/home/ubuntu/data/")


cwd = os.getcwd()
if 'ubuntu' in cwd:
    project_path = os.getcwd()+'/'
    data_path = '/home/ubuntu/data/'
    print('ProjectData project_path = ',cwd)
else:
    project_path = 'path'


import os
os.getcwd()
device = torch.device('cuda')

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x.cpu() * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
    x = x.transpose(1, 3)
    return x


def custom_replace_normalise(tensor, on_zero, on_non_zero):
    """ Create Tensor with binary values
        Rule: Non Zero
    """
    res = tensor.clone()
    res = res * 0.5 + 0.5
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

def custom_replace(tensor, on_zero, on_non_zero):
    """ Create Tensor with binary values
        Rule: Non Zero
    """
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

def evaluate(model, dataset, filename, GPU_YN_IN=False):
    if GPU_YN_IN==True:
        criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)
    else:
        criterion = InpaintingLoss(VGG16FeatureExtractor())


    #image, mask, gt = zip(*[dataset[i] for i in range(8)])

    test_set_loader = torch.utils.data.DataLoader(dataset,
                                                  # batch_size=len(dataset),
                                                  batch_size=1,
                                                  shuffle=True)
    data_iterator_test = iter(test_set_loader)

    # DEAL WITH MEMORY ERRORS
    counter=0
    test_loss = 0
    for i in data_iterator_test:
        counter+=1
        if counter <17:

            if GPU_YN_IN==True:
                # gt, label, image, mask, pca = [x.to(device) for x in next(data_iterator_test)]
                gt, label, image, mask, pca = [x for x in next(data_iterator_test)]
            else:
                gt, label, image, mask, pca = [x for x in next(data_iterator_test)]

            gt = gt.cuda()
            image = image.cuda()
            mask = mask.cuda()
            pca = pca.cuda()
            gt_mask = custom_replace(gt, 0, 1)


            with torch.no_grad():
                # model = PConvUNet().to(device)
                output, _ = model(pca, gt_mask)
                # loss_dict = criterion(image, mask, output, gt) #added
                loss_dict = criterion(pca, gt_mask, output, gt)

                test_loss = 0
                for key, coef in opt.LAMBDA_DICT.items():
                    value = coef * loss_dict[key]
                    test_loss += value

                test_loss = test_loss/counter

            # full_mask = custom_replace(image,0,1)
            # output_comp = full_mask * image + (1 - full_mask) * output 
            output_comp = gt_mask * gt + (1 - gt_mask) * output

            try:
                pca_concat
                pca_concat = torch.cat((pca_concat, pca.data))
                image_concat = torch.cat((image_concat, image.data))
                gt_mask_concat = torch.cat((gt_mask_concat, gt_mask.data))
                output_comp_concat = torch.cat((output_comp_concat, output_comp.data))
                gt_concat = torch.cat((gt_concat, gt.data))
                output_concat = torch.cat((output_concat, output.data))

                label_concat = label_concat +[label[0]]

            except NameError:
                pca_concat = pca.data
                gt_mask_concat = gt_mask.data
                output_concat = output.data
                output_comp_concat = output_comp.data
                image_concat = image.data
                gt_concat = gt.data
                # label_concat = list()

                label_concat = [label[0]]
                # print('type label_concat:',type(label_concat))
            # print('label_concat',label_concat)


    # output_comp = (1 - mask) * image + mask * output
    # output_comp = output_comp * 0.5 + 0.5

    if GPU_YN_IN==True:
        grid = make_grid(

        torch.cat((gt_concat, pca_concat, output_concat,gt_mask_concat ), dim=0)) 


        save_image(grid, filename)
    else:
        grid = make_grid(

            torch.cat((pca, gt_mask.cpu(), output,
                        output_comp, gt,gt_mask), dim=0))
        save_image(grid, filename)
    print('EVAL COMPLETE')

    # Epochs are finished
    if os.path.isdir(project_path+'models/output_imgs/')==False:
        os.mkdir(project_path+'models/output_imgs/')

    # gt, _, image, mask, pca = [x.to(device) for x in next(iterator_train)]

    #Store final images
    norm = 'Y'
    output_concat = output_concat.reshape(-1,3,128,128)
    for i, i_name in zip(output_concat,label_concat):
        save_image(i, project_path+'models/output_imgs/'+i_name)


    del pca_concat,gt_mask_concat,output_concat,output_comp_concat,image_concat,gt_concat,output, _
    return test_loss
