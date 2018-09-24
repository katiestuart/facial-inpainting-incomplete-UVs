import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            print('ds')
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        # Add back in BatchNorm & Leaky!!!
        
        self.e_1 = nn.Conv2d(channels, 64, 4, stride=2, padding=1)
        # self.b1  = nn.BatchNorm2d(64, 0.8),nn.LeakyReLU(0.2)
        # self.l1  = nn.LeakyReLU(0.2)

        self.e_2 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        # self.b2  = nn.BatchNorm2d(64, 0.8),nn.LeakyReLU(0.2)
        # self.l1  = nn.LeakyReLU(0.2)

        self.e_3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        # self.b3  = nn.BatchNorm2d(128, 0.8),nn.LeakyReLU(0.2)
        # self.l3  = nn.LeakyReLU(0.2)

        self.e_4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        # self.b4  = nn.BatchNorm2d(128, 0.8),nn.LeakyReLU(0.2)
        # self.l4  = nn.LeakyReLU(0.2)

        self.e_5 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        # self.b5  = nn.BatchNorm2d(128, 0.8),nn.LeakyReLU(0.2)
        # self.l5  = nn.LeakyReLU(0.2)

        self.e_6 = nn.Conv2d(512, 4000, 1)

        self.ed_1 = nn.ConvTranspose2d(4000, 512, 4, stride=2, padding=1)
        # self.bd1  = nn.BatchNorm2d(512, 0.8),nn.LeakyReLU(0.2)
        # self.ld1  = nn.LeakyReLU(0.2)

        self.ed_2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        # self.bd2  = nn.BatchNorm2d(256, 0.8),nn.LeakyReLU(0.2)
        # self.ld1  = nn.LeakyReLU(0.2)

        self.ed_3 = nn.ConvTranspose2d(256, 128, 4, stride=1, padding=1)
        # self.bd3  = nn.BatchNorm2d(128, 0.8),nn.LeakyReLU(0.2)
        # self.ld3  = nn.LeakyReLU(0.2)

        self.ed_4 = nn.ConvTranspose2d(128, 64, 4, stride=2  , padding=2)
        # self.bd4  = nn.BatchNorm2d(64, 0.8),nn.LeakyReLU(0.2)
        # self.ld4  = nn.LeakyReLU(0.2)

        self.ed_5 = nn.Conv2d(64, channels, 3, 1, 1)


    def forward(self, x):
                    x = self.e_1(x)

                    x = self.e_2(x)

                    x = self.e_3(x)

                    x = self.e_4(x)

                    x = self.e_5(x)

                    x = self.e_6(x)

                    x = self.ed_1(x)

                    x = self.ed_2(x)

                    x = self.ed_3(x)

                    x = self.ed_4(x)

                    x = self.ed_5(x)

                    return x


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [ (64, 2, False),
                                                (128, 2, True),
                                                (256, 1, True),
                                                (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
