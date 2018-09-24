import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]




class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv, self).__init__()

        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                            stride, padding, dilation, groups, bias)

        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        #
        # up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.input_conv = up(self.input_conv)
        # self.mask_conv = up(self.mask_conv)

        #self.input_conv = F.upsample(self.input_conv, scale_factor=2)
        # self.input_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     )
        #
        # self.mask_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     )

        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False, dilation=3):
        super(PCBActiv, self).__init__()
        #in_channels, out_channels, kernel_size,stride, padding, dilation, groups, bias
        if sample == 'layer-1':
            # print("layer 1 Check!")
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias, dilation= dilation)

        elif sample == 'layer-2':
            # print("layer 2 Check!")
            self.conv = PartialConv(in_ch, out_ch, 12, 3, 0, bias=conv_bias, dilation= dilation)

        elif sample == 'layer-3':
            # print("layer 3 Check!")
            self.conv = PartialConv(in_ch, out_ch, 9, 1, 0, bias=conv_bias, dilation= dilation)

        elif sample == 'layer-4':
            # print("layer 4 Check!")
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 4, bias=conv_bias, dilation= dilation)

        elif sample == 'layer-5':
            # print("layer 5 Check!")
            self.conv = PartialConv(in_ch, out_ch, 1, 4, 0, bias=conv_bias, dilation= dilation)

        elif sample == 'layer-6':
            # print("layer 6 Check!")
            self.conv = PartialConv(in_ch, out_ch, 1, 4, 0, bias=conv_bias, dilation= dilation)

        elif sample == 'layer-7':
            # print("layer 6 Check!")
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias, dilation= dilation)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias, dilation= dilation)
            # print("Other Layer Check!")

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask



class PConvUNet(nn.Module):
    def __init__(self, layer_size=7):
        super(PConvUNet, self).__init__()
        self.freeze_enc_bn = False
        # Brilliant - Also works for fewer layers
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(3,   64, bn=False, sample='layer-1', dilation = 1)
        self.enc_2 = PCBActiv(64,  128, sample='layer-2', dilation = 3)
        self.enc_3 = PCBActiv(128, 256, sample='layer-3', dilation = 6)
        self.enc_4 = PCBActiv(256, 512, sample='layer-4', dilation = 12)
        self.enc_5 = PCBActiv(512, 512, sample='layer-5', dilation = 18)
        self.enc_6 = PCBActiv(512, 512, sample='layer-6', dilation = 24)

        for i in range(6, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='layer-7', dilation = 1))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky', dilation = 1))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky', dilation = 1)
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky', dilation = 1)
        self.dec_2 = PCBActiv(128 + 64 , 64 , activ='leaky', dilation = 1)
        self.dec_1 = PCBActiv(64 + 3, 3, bn=False, activ=None, conv_bias=True, dilation = 1)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask


        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            # print('--------------------')
            # print("ENC h_key:", h_key)
            # print("ENC l_key:", l_key)
            # print("ENC h:h_dict[h_key]",h_dict[h_key].size())

            h_key_prev = h_key

            if h_key in ['h_1','h_2','h_3','h_4','h_5','h_6']:
                h_dict[h_key] = F.upsample(h_dict[h_key], scale_factor=2)
                h_mask_dict[h_key] = F.upsample(h_mask_dict[h_key], scale_factor=2)
                # print("ENC post_upsample:", h_key)
                # print('ENC post_upsample: h_dict[h_key]', h_dict[h_key].size())

        h_key = 'h_{:d}'.format(self.layer_size)
        # print('----------------------------------------')
        # print("MID h_key:", h_key)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]
        # print("MID h.size()", h.size())
        # print('----------------------------------------')


        # adds the DECODER layers to the dict
        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            ## included in the original
            if enc_h_key!='h_0':
                h = F.upsample(h, scale_factor=2)
                h_mask = F.upsample(h_mask, scale_factor=2)

            # print("DEC enc_h_key:", enc_h_key)
            # print("DEC enc_h_key:", dec_l_key)
            # print("DEC h.size()",h.size())

            # print("DEC h_dict[enc_h_key]:", h_dict[enc_h_key].size())
            # print('--------------------')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(PConvUNet, self).train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()




