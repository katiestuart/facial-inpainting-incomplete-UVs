import torch.nn as nn
import torch.nn.functional as F
import torch

debug_yn = 'N'

class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        # def downsample(in_feat, out_feat, normalize=True):
        #     layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
        #     if normalize:
        #         layers.append(nn.BatchNorm2d(out_feat, 0.8))
        #     layers.append(nn.LeakyReLU(0.2))
        #     print('ds')
        #     return layers
        #
        # def upsample(in_feat, out_feat, normalize=True):
        #     layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
        #     if normalize:
        #         layers.append(nn.BatchNorm2d(out_feat, 0.8))
        #     layers.append(nn.ReLU())
        #     return layers
        #
        # self.model = nn.Sequential(
        #     nn.Conv2d(channels, 64, 4, stride=2, padding=1),nn.LeakyReLU(0.2),
        #     nn.Conv2d(64, 64, 4, stride=2, padding=1),nn.BatchNorm2d(64, 0.8),nn.LeakyReLU(0.2),
        #     nn.Conv2d(64, 128, 4, stride=2, padding=1),nn.BatchNorm2d(128, 0.8),nn.LeakyReLU(0.2),
        #     nn.Conv2d(128, 256, 4, stride=2, padding=1),nn.BatchNorm2d(256, 0.8),nn.LeakyReLU(0.2),
        #     nn.Conv2d(256, 512, 4, stride=2, padding=1),nn.BatchNorm2d(512, 0.8),nn.LeakyReLU(0.2),
        #     nn.Conv2d(512, 4000, 1),
        #     nn.ConvTranspose2d(4000, 512, 4, stride=2, padding=1),nn.BatchNorm2d(512, 0.8),nn.ReLU(),
        #     nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),nn.BatchNorm2d(256, 0.8),nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),nn.BatchNorm2d(128, 0.8),nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),nn.BatchNorm2d(64, 0.8),nn.ReLU(),
        #     # nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),nn.BatchNorm2d(32, 0.8),nn.ReLU(),
        #     nn.Conv2d(64, channels, 3, 1, 1),
        #     nn.Tanh())



        self.e_1 = nn.Conv2d(channels, 64, 4, stride=2, padding=1)
        # self.b1  = nn.BatchNorm2d(64, 0.8),nn.LeakyReLU(0.2)
        self.l1  = nn.LeakyReLU(0.2)

        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64, 0.8),nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2))

        self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128, 0.8),nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2))

        self.layer4 = nn.Sequential(
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256, 0.8),nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2))

        self.layer5 = nn.Sequential(
                nn.Conv2d(256, 512, 4, stride=2, padding=1),
                nn.BatchNorm2d(512, 0.8),nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2))

        self.layerd_5 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256, 0.8),nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2))

        self.layerd_4 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128, 0.8),nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2))

        self.layerd_3 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64, 0.8),nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2))

        self.layerd_2 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, stride=4, padding=0),
                nn.BatchNorm2d(64, 0.8),nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2))




        self.layerd_1 = nn.Sequential(
                nn.Conv2d(64, channels, 3, 1, 1),
                nn.Tanh())

    def forward(self, x):
                        # print('input',x.shape)
                        x = self.e_1(x)
                        # x = self.b1(x)
                        x = self.l1(x)
                        # print("e1:", x.shape)
                        x = self.layer2(x)
                        x = self.layer3(x)
                        x = self.layer4(x)
                        x = self.layer5(x)
                        x = self.layerd_5(x)
                        x = self.layerd_4(x)
                        x = self.layerd_3(x)
                        x = self.layerd_2(x)
                        x = self.layerd_1(x)


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
        for out_filters, stride, normalize in [ (64, 1, False),
                                                (128, 1, True),
                                                (256, 1, True),
                                                (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
