import torch
import torch.nn as nn
import numpy as np
from model import DeepLayerAggregationNetWithResnet

from layers import ConvLayer
from layers import ResidualBlock

"""This script implements a small model and experiments with 
how to initialize weights and biases of layers in a model"""

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 3, 3)
#         self.conv2 = nn.ConvTranspose2d(3, 3, 3)
#         self.fc1 = nn.Linear(3, 3)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = torch.relu(self.fc1(x))
#         return x


# model = ConvLayer(10, 3, initialize=True)

# model = ResidualBlock(1, downsample=False, initialize=True)
#
# print(list(model.parameters()))

# weight = list(model.parameters())

# print(torch.amax(weight[-2]))
# print(torch.amin(weight[-2]))
# print(torch.mean(weight[-2]))
# print(torch.std(weight[-2]))

# print(np.sqrt(0.02/3/3/10))

# tensor(0.0958, grad_fn=<AmaxBackward0>)
# tensor(-0.0960, grad_fn=<AminBackward0>)
# tensor(0.0001, grad_fn=<MeanBackward0>)
# tensor(0.0552, grad_fn=<StdBackward0>)

# tensor(0.3906, grad_fn=<AmaxBackward0>)
# tensor(-0.4951, grad_fn=<AminBackward0>)
# tensor(-0.0031, grad_fn=<MeanBackward0>)
# tensor(0.1349, grad_fn=<StdBackward0>)


# def parameter_initializer(layer):
#     if isinstance(layer, nn.Conv2d):
#         nn.init.zeros_(layer.bias.data)
#     if isinstance(layer, nn.ConvTranspose2d):
#         nn.init.zeros_(layer.bias.data)
#     if isinstance(layer, nn.Conv2d):
#         nn.init.zeros_(layer.weight.data)
# model.apply(parameter_initializer)
#
# print(list(model.parameters()))