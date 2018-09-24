import torch
import torch.nn as nn
from torch.autograd import Variable

def custom_replace_normalise(tensor, on_zero, on_non_zero):
    """ Create Tensor with binary values
        Rule: Non Zero
    """
    res = tensor.clone()
    # res = res * 0.5 + 0.5
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res


def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


def weighted_mse_loss(input, target, weights):
    """ L2 Loss
        Currently not accounting for number of non-zero pixels
        I.e images that have a small number of non-zero https://arxiv.org/pdf/1704.04086.pdf
    """
    out = (input-target)**2
    out = out * weights
    return out.sum()/input.data.nelement()


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super(InpaintingLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        # pca, gt_mask, output, gt
        full_mask = custom_replace_normalise(input,0,1)
        full_mask_gt = custom_replace_normalise(gt,0,1)

        loss_dict = {}
        # output_comp = mask * input + (1 - mask) * output
        output_comp = full_mask * input + (1 - full_mask) * output  ###### add in original mask here from PCA.

        #loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        hole_mask =  (1 - mask)
        loss_dict['hole'] = weighted_mse_loss(output,gt,full_mask_gt)


        # loss_dict['valid'] = self.l1(mask * output, mask * gt)
        # loss_dict['valid'] = self.l1(full_mask * output, full_mask * gt)  ###### add in original mask here from PCA.
        loss_dict['valid'] = weighted_mse_loss(output,input,full_mask)


        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)

        # perceptual
        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        error_sq = (input-gt)**2
        mse = float(error_sq.sum())/input.data.nelement()
        mse = torch.Tensor([mse])
        loss_dict['mse'] = mse

        # print('loss debug')
        # print(input[0])
        # print(error_sq[0])
        # print(input.data.nelement())
        # print(error_sq.sum())
        # print(float(error_sq.sum()))
        # print(error_sq.sum()/input.data.nelement())
        # print(float(error_sq.sum())/input.data.nelement())
        # print(loss_dict)
        # print(torch.Tensor(dr))

        return loss_dict


