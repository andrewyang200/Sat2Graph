import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


"""This script compares the differences between the Transpose Convolution layer between Tensorflow and Pytorch.
We found that the difference lies in how the output padding was applied. PyTorch applies output padding on the 
bottom and right side of the input tensor. TF applies output padding on the top and left. 
Note: Both Pytorch and TF flips the kernel both vertically and horizontally"""


# kernel_size = 3
# stride = 2
# padding = kernel_size // 2
# output_padding = 0
#
# kernel_1_weights = {
#     'weight': torch.tensor([[[[0.28366318]]]]),
#     'bias': torch.tensor([0.])
# }
#
# kernel_3_weights = {
#     '1.weight': torch.permute(torch.tensor([[[[-0.5013187]], [[-0.3087212]],[[-0.0520433]]], [[[0.21473248]], [[-0.48538214]], [[1.4213867]]], [[[-0.19337486]], [[-1.306662]], [[-0.7508448]]]]), (3, 2, 0, 1)),
#     '1.bias': torch.tensor([0.])
# }
#
# torch.save(kernel_3_weights, "kernel_3_weights.pt")
#
# x = torch.tensor([[[[2.0, 2.0], [2.0, 2.0]]]])
# # x = torch.tensor([[[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]]
#
# model = nn.Sequential(
#     nn.ZeroPad2d((1, 0, 1, 0)),
#     nn.ConvTranspose2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
# )
# # model.load_state_dict(torch.load("kernel_1_weights.pt"))
# model.load_state_dict(torch.load("kernel_3_weights.pt"))
#
# output = model(x)
# # print(output)
# #
# # print(torch.mean(output))
#
# print(kernel_3_weights['1.weight'] *2)
#
# print(output)
#
# print(output[:, :, 1:, 1:])
