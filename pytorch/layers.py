import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_bias=True, downsample=False, initialize=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(channels, momentum=0.01)
        self.relu1 = nn.ReLU()

        if self.downsample:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=use_bias)
            self.conv_init = nn.Conv2d(channels, channels, kernel_size=1, stride=2, bias=use_bias)
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=use_bias)

        self.bn2 = nn.BatchNorm2d(channels, momentum=0.01)

        self.relu2 = nn.ReLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=use_bias)

        if initialize:
            if self.downsample:
                nn.init.kaiming_normal_(self.conv_init.weight, nonlinearity='relu')
                nn.init.zeros_(self.conv_init.bias)
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv1.bias)
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        x_init = x
        out = self.bn1(x)
        out = self.relu1(out)

        out = self.conv1(out)
        if self.downsample:
            x_init = self.conv_init(x_init)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += x_init

        return out


# not used in the current Sat2Graph model
class BottleneckBlock(nn.Module):
    def __init__(self, channels, use_bias=True, downsample=False):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(channels, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.01)

        if self.downsample:
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=use_bias)
            self.conv_sc = nn.Conv2d(channels, channels*4, kernel_size=1, stride=2, bias=use_bias)
        else:
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=use_bias)
            self.conv_sc = nn.Conv2d(channels, channels*4, kernel_size=1, stride=1, bias=use_bias)

        self.bn3 = nn.BatchNorm2d(channels, momentum=0.01)

        self.conv3 = nn.Conv2d(channels, channels*4, kernel_size=1, stride=1, bias=use_bias)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)

        shortcut = out

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        shortcut = self.conv_sc(shortcut)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += shortcut

        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', kernel_size=3, stride=2, batchnorm=False, add=None, deconv=False, output_padding=None, initialize=False):
        super(ConvLayer, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.add = add
        self.deconv = deconv

        if output_padding is None:
            self.output_padding = stride-1
        else:
            self.output_padding = output_padding

        if not deconv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=kernel_size//2, output_padding=self.output_padding)
        if initialize:
            std = np.sqrt(0.02 / kernel_size / kernel_size / in_channels)
            nn.init.trunc_normal_(self.conv.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            nn.init.zeros_(self.conv.bias)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        if activation == 'relu':
            self.ac = nn.ReLU()
        elif activation == 'sigmoid':
            self.ac = nn.Sigmoid()
        elif activation == 'tanh':
            self.ac = nn.Tanh()
        elif activation == 'linear':
            self.ac = None

    def forward(self, x):
        x = self.conv(x)

        if self.add is not None:
            x = x + self.add

        if self.batchnorm:
            x = self.bn(x)

        if self.ac is not None:
            x = self.ac(x)

        return x


class TFConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', kernel_size=3, stride=2, batchnorm=False, add=None, deconv=False, initialize=False):
        super(TFConvLayer, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.add = add
        self.deconv = deconv

        if not deconv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        else:
            # manually adds output padding to the top of left of the input image to match
            self.padding = nn.ZeroPad2d((1, 0, 1, 0))
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        if initialize:
            std = np.sqrt(0.02 / kernel_size / kernel_size / in_channels)
            nn.init.trunc_normal_(self.conv.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            nn.init.zeros_(self.conv.bias)

        if activation == 'relu':
            self.ac = nn.ReLU()
        elif activation == 'sigmoid':
            self.ac = nn.Sigmoid()
        elif activation == 'tanh':
            self.ac = nn.Tanh()
        elif activation == 'linear':
            self.ac = None

    def forward(self, x):
        if self.deconv:
            x = self.padding(x)

        x = self.conv(x)
        x = x[:, :, 1:, 1:]

        if self.add is not None:
            x = x + self.add

        if self.batchnorm:
            x = self.bn(x)

        if self.ac is not None:
            x = self.ac(x)

        return x

