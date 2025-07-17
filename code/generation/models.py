import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self,
                 input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()
        encoder_lis = [
            # 输入:3*112*112
            nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64*112*112
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 128*56*56
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            #256*28*28
        ]

        bottle_neck_lis = [ResnetBlock(256),
                            ResnetBlock(256),
                            ResnetBlock(256),
                           ]

        decoder_lis = [
            #输入256*28*28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([128,56,56]),
            nn.ReLU(),
            # state size. 128 x 56 x 56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([64,112,112]),
            nn.ReLU(),
            # state size. 64 x 112 x 112
            nn.ConvTranspose2d(64, image_nc, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
            # state size. 3 x 112 x 112
        ]

     #  self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # 人脸图片: 3*112*112
        model = [
            nn.Conv2d(image_nc, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.995),
            #nn.LeakyReLU(0.2),
            # 32*56*56
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64,eps=0.001,momentum=0.995),
            #nn.LeakyReLU(0.2),
            # 64*28*28
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128,eps=0.001,momentum=0.995),
            nn.LeakyReLU(0.2),
            # 128*14*14
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256,eps=0.001,momentum=0.995),
            nn.LeakyReLU(0.2),
            # 256*7*7
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512,eps=0.001,momentum=0.995),
            nn.LeakyReLU(0.2),
            # 512*4*4
            nn.Conv2d(512, 1, 1)
            #nn.Sigmoid()            #WGAN

        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x)
        output = torch.reshape(output,[-1,1]).squeeze()
        return output

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

