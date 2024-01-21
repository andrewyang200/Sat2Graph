import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from subprocess import Popen
import numpy as np
import random
import scipy.ndimage
import matplotlib.pyplot as plt
import pickle
import PIL.Image as Image
import json
from model import Sat2Graph

import layers
from layers import TFConvLayer
from model import ResnetBlock
from model import ReduceBlock
from model import AggregateBlock
from layers import ResidualBlock


"""This script breaks apart individual layers in the sat2graph model and tests for differences between 
Tensorflow's implementation and Pytorch's implementation"""


# p = "/Users/andrewyang/PycharmProjects/model_infer_tests/infer_test_samples.pickle"
# data = pickle.load(open(p, "rb"), encoding='latin1')
#
# weights = torch.load(open("sat2g_tf_weights_state_dict.pt", 'rb'))
#
# conv1_weights = {
#     "conv.weight": weights["model.conv1.conv.weight"],
#     "conv.bias": weights["model.conv1.conv.bias"]
# }
#
# conv2_weights = {
#     "0.conv.weight": weights["model.conv1.conv.weight"],
#     "0.conv.bias": weights["model.conv1.conv.bias"],
#     "1.conv.weight": weights["model.conv2.conv.weight"],
#     "1.conv.bias": weights["model.conv2.conv.bias"],
#     "1.bn.weight": weights["model.conv2.bn.weight"],
#     "1.bn.bias": weights["model.conv2.bn.bias"],
#     "1.bn.running_mean": weights["model.conv2.bn.running_mean"],
#     "1.bn.running_var": weights["model.conv2.bn.running_var"],
#     "1.bn.num_batches_tracked": weights["model.conv2.bn.num_batches_tracked"]
# }
#
# reduce_block_weights = {
#     "0.conv.weight": weights["model.conv1.conv.weight"],
#     "0.conv.bias": weights["model.conv1.conv.bias"],
#     "1.conv.weight": weights["model.conv2.conv.weight"],
#     "1.conv.bias": weights["model.conv2.conv.bias"],
#     "1.bn.weight": weights["model.conv2.bn.weight"],
#     "1.bn.bias": weights["model.conv2.bn.bias"],
#     "1.bn.running_mean": weights["model.conv2.bn.running_mean"],
#     "1.bn.running_var": weights["model.conv2.bn.running_var"],
#     "1.bn.num_batches_tracked": weights["model.conv2.bn.num_batches_tracked"],
#     "2.conv1.conv.weight": weights["model.r_4s.conv1.conv.weight"],
#     "2.conv1.conv.bias": weights["model.r_4s.conv1.conv.bias"],
#     "2.conv1.bn.weight": weights["model.r_4s.conv1.bn.weight"],
#     "2.conv1.bn.bias": weights["model.r_4s.conv1.bn.bias"],
#     "2.conv1.bn.running_mean": weights["model.r_4s.conv1.bn.running_mean"],
#     "2.conv1.bn.running_var": weights["model.r_4s.conv1.bn.running_var"],
#     "2.conv1.bn.num_batches_tracked": weights["model.r_4s.conv1.bn.num_batches_tracked"],
#     "2.conv2.conv.weight": weights["model.r_4s.conv2.conv.weight"],
#     "2.conv2.conv.bias": weights["model.r_4s.conv2.conv.bias"],
#     "2.conv2.bn.weight": weights["model.r_4s.conv2.bn.weight"],
#     "2.conv2.bn.bias": weights["model.r_4s.conv2.bn.bias"],
#     "2.conv2.bn.running_mean": weights["model.r_4s.conv2.bn.running_mean"],
#     "2.conv2.bn.running_var": weights["model.r_4s.conv2.bn.running_var"],
#     "2.conv2.bn.num_batches_tracked": weights["model.r_4s.conv2.bn.num_batches_tracked"]
# }
#
# resnet_block_weights = {
#     "0.conv.weight": weights["model.conv1.conv.weight"],
#     "0.conv.bias": weights["model.conv1.conv.bias"],
#     "1.conv.weight": weights["model.conv2.conv.weight"],
#     "1.conv.bias": weights["model.conv2.conv.bias"],
#     "1.bn.weight": weights["model.conv2.bn.weight"],
#     "1.bn.bias": weights["model.conv2.bn.bias"],
#     "1.bn.running_mean": weights["model.conv2.bn.running_mean"],
#     "1.bn.running_var": weights["model.conv2.bn.running_var"],
#     "1.bn.num_batches_tracked": weights["model.conv2.bn.num_batches_tracked"],
#     "2.conv1.conv.weight": weights["model.r_4s.conv1.conv.weight"],
#     "2.conv1.conv.bias": weights["model.r_4s.conv1.conv.bias"],
#     "2.conv1.bn.weight": weights["model.r_4s.conv1.bn.weight"],
#     "2.conv1.bn.bias": weights["model.r_4s.conv1.bn.bias"],
#     "2.conv1.bn.running_mean": weights["model.r_4s.conv1.bn.running_mean"],
#     "2.conv1.bn.running_var": weights["model.r_4s.conv1.bn.running_var"],
#     "2.conv1.bn.num_batches_tracked": weights["model.r_4s.conv1.bn.num_batches_tracked"],
#     "2.conv2.conv.weight": weights["model.r_4s.conv2.conv.weight"],
#     "2.conv2.conv.bias": weights["model.r_4s.conv2.conv.bias"],
#     "2.conv2.bn.weight": weights["model.r_4s.conv2.bn.weight"],
#     "2.conv2.bn.bias": weights["model.r_4s.conv2.bn.bias"],
#     "2.conv2.bn.running_mean": weights["model.r_4s.conv2.bn.running_mean"],
#     "2.conv2.bn.running_var": weights["model.r_4s.conv2.bn.running_var"],
#     "2.conv2.bn.num_batches_tracked": weights["model.r_4s.conv2.bn.num_batches_tracked"],
#     "3.resnet.0.bn1.weight": weights["model.n_4s.resnet.0.bn1.weight"],
#     "3.resnet.0.bn1.bias": weights["model.n_4s.resnet.0.bn1.bias"],
#     "3.resnet.0.bn1.running_mean": weights["model.n_4s.resnet.0.bn1.running_mean"],
#     "3.resnet.0.bn1.running_var": weights["model.n_4s.resnet.0.bn1.running_var"],
#     "3.resnet.0.bn1.num_batches_tracked": weights["model.n_4s.resnet.0.bn1.num_batches_tracked"],
#     "3.resnet.0.conv1.weight": weights["model.n_4s.resnet.0.conv1.weight"],
#     "3.resnet.0.conv1.bias": weights["model.n_4s.resnet.0.conv1.bias"],
#     "3.resnet.0.bn2.weight": weights["model.n_4s.resnet.0.bn2.weight"],
#     "3.resnet.0.bn2.bias": weights["model.n_4s.resnet.0.bn2.bias"],
#     "3.resnet.0.bn2.running_mean": weights["model.n_4s.resnet.0.bn2.running_mean"],
#     "3.resnet.0.bn2.running_var": weights["model.n_4s.resnet.0.bn2.running_var"],
#     "3.resnet.0.bn2.num_batches_tracked": weights["model.n_4s.resnet.0.bn2.num_batches_tracked"],
#     "3.resnet.0.conv2.weight": weights["model.n_4s.resnet.0.conv2.weight"],
#     "3.resnet.0.conv2.bias": weights["model.n_4s.resnet.0.conv2.bias"],
#     "3.bn.weight": weights["model.n_4s.bn.weight"],
#     "3.bn.bias": weights["model.n_4s.bn.bias"],
#     "3.bn.running_mean": weights["model.n_4s.bn.running_mean"],
#     "3.bn.running_var": weights["model.n_4s.bn.running_var"],
#     "3.bn.num_batches_tracked": weights["model.n_4s.bn.num_batches_tracked"]
# }
#
# residual_block_weights = {
#     "0.conv.weight": weights["model.conv1.conv.weight"],
#     "0.conv.bias": weights["model.conv1.conv.bias"],
#     "1.conv.weight": weights["model.conv2.conv.weight"],
#     "1.conv.bias": weights["model.conv2.conv.bias"],
#     "1.bn.weight": weights["model.conv2.bn.weight"],
#     "1.bn.bias": weights["model.conv2.bn.bias"],
#     "1.bn.running_mean": weights["model.conv2.bn.running_mean"],
#     "1.bn.running_var": weights["model.conv2.bn.running_var"],
#     "1.bn.num_batches_tracked": weights["model.conv2.bn.num_batches_tracked"],
#     "2.conv1.conv.weight": weights["model.r_4s.conv1.conv.weight"],
#     "2.conv1.conv.bias": weights["model.r_4s.conv1.conv.bias"],
#     "2.conv1.bn.weight": weights["model.r_4s.conv1.bn.weight"],
#     "2.conv1.bn.bias": weights["model.r_4s.conv1.bn.bias"],
#     "2.conv1.bn.running_mean": weights["model.r_4s.conv1.bn.running_mean"],
#     "2.conv1.bn.running_var": weights["model.r_4s.conv1.bn.running_var"],
#     "2.conv1.bn.num_batches_tracked": weights["model.r_4s.conv1.bn.num_batches_tracked"],
#     "2.conv2.conv.weight": weights["model.r_4s.conv2.conv.weight"],
#     "2.conv2.conv.bias": weights["model.r_4s.conv2.conv.bias"],
#     "2.conv2.bn.weight": weights["model.r_4s.conv2.bn.weight"],
#     "2.conv2.bn.bias": weights["model.r_4s.conv2.bn.bias"],
#     "2.conv2.bn.running_mean": weights["model.r_4s.conv2.bn.running_mean"],
#     "2.conv2.bn.running_var": weights["model.r_4s.conv2.bn.running_var"],
#     "2.conv2.bn.num_batches_tracked": weights["model.r_4s.conv2.bn.num_batches_tracked"],
#     "3.bn1.weight": weights["model.n_4s.resnet.0.bn1.weight"],
#     "3.bn1.bias": weights["model.n_4s.resnet.0.bn1.bias"],
#     "3.bn1.running_mean": weights["model.n_4s.resnet.0.bn1.running_mean"],
#     "3.bn1.running_var": weights["model.n_4s.resnet.0.bn1.running_var"],
#     "3.bn1.num_batches_tracked": weights["model.n_4s.resnet.0.bn1.num_batches_tracked"],
#     "3.conv1.weight": weights["model.n_4s.resnet.0.conv1.weight"],
#     "3.conv1.bias": weights["model.n_4s.resnet.0.conv1.bias"],
#     "3.bn2.weight": weights["model.n_4s.resnet.0.bn2.weight"],
#     "3.bn2.bias": weights["model.n_4s.resnet.0.bn2.bias"],
#     "3.bn2.running_mean": weights["model.n_4s.resnet.0.bn2.running_mean"],
#     "3.bn2.running_var": weights["model.n_4s.resnet.0.bn2.running_var"],
#     "3.bn2.num_batches_tracked": weights["model.n_4s.resnet.0.bn2.num_batches_tracked"],
#     "3.conv2.weight": weights["model.n_4s.resnet.0.conv2.weight"],
#     "3.conv2.bias": weights["model.n_4s.resnet.0.conv2.bias"]
# }
#
# bn_weights = {
#     "weight": weights["model.n_4s.bn.weight"],
#     "bias": weights["model.n_4s.bn.bias"],
#     "running_mean": weights["model.n_4s.bn.running_mean"],
#     "running_var": weights["model.n_4s.bn.running_var"],
#     "num_batches_tracked": weights["model.n_4s.bn.num_batches_tracked"]
# }
#
# aggregate_block_weights = {
#     # [input, output, kernel, kernel]
#     "conv1.conv.weight": weights["model.a1_2s.conv1.conv.weight"],
#     "conv1.conv.bias": weights["model.a1_2s.conv1.conv.bias"],
#     "conv1.bn.weight": weights["model.a1_2s.conv1.bn.weight"],
#     "conv1.bn.bias": weights["model.a1_2s.conv1.bn.bias"],
#     "conv1.bn.running_mean": weights["model.a1_2s.conv1.bn.running_mean"],
#     "conv1.bn.running_var": weights["model.a1_2s.conv1.bn.running_var"],
#     "conv1.bn.num_batches_tracked": weights["model.a1_2s.conv1.bn.num_batches_tracked"],
#     "conv2.conv.weight": weights["model.a1_2s.conv2.conv.weight"],
#     "conv2.conv.bias": weights["model.a1_2s.conv2.conv.bias"],
#     "conv2.bn.weight": weights["model.a1_2s.conv2.bn.weight"],
#     "conv2.bn.bias": weights["model.a1_2s.conv2.bn.bias"],
#     "conv2.bn.running_mean": weights["model.a1_2s.conv2.bn.running_mean"],
#     "conv2.bn.running_var": weights["model.a1_2s.conv2.bn.running_var"],
#     "conv2.bn.num_batches_tracked": weights["model.a1_2s.conv2.bn.num_batches_tracked"],
#     "conv3.conv.weight": weights["model.a1_2s.conv3.conv.weight"],
#     "conv3.conv.bias": weights["model.a1_2s.conv3.conv.bias"],
#     "conv3.bn.weight": weights["model.a1_2s.conv3.bn.weight"],
#     "conv3.bn.bias": weights["model.a1_2s.conv3.bn.bias"],
#     "conv3.bn.running_mean": weights["model.a1_2s.conv3.bn.running_mean"],
#     "conv3.bn.running_var": weights["model.a1_2s.conv3.bn.running_var"],
#     "conv3.bn.num_batches_tracked": weights["model.a1_2s.conv3.bn.num_batches_tracked"]
# }
#
#
#
# torch.save(aggregate_block_weights, "aggregate_block_weights.pt")
#
#
# conv1 = layers.ConvLayer(3, 12, kernel_size=5, stride=1, batchnorm=False)
#
# conv2 = nn.Sequential(
#     layers.ConvLayer(3, 12, kernel_size=5, stride=1, batchnorm=False),
#     layers.ConvLayer(12, 24, kernel_size=5, stride=2, batchnorm=True)
# )
#
# reduce_block = nn.Sequential(
#     layers.ConvLayer(3, 12, kernel_size=5, stride=1, batchnorm=False),
#     layers.ConvLayer(12, 24, kernel_size=5, stride=2, batchnorm=True),
#     ReduceBlock(24, 48)
# )
#
# resnet_block = nn.Sequential(
#     layers.ConvLayer(3, 12, kernel_size=5, stride=1, batchnorm=False),
#     layers.ConvLayer(12, 24, kernel_size=5, stride=2, batchnorm=True),
#     ReduceBlock(24, 48),
#     ResnetBlock(48, resnet_step=1)
# )
#
# residual_block = nn.Sequential(
#     layers.ConvLayer(3, 12, kernel_size=5, stride=1, batchnorm=False),
#     layers.ConvLayer(12, 24, kernel_size=5, stride=2, batchnorm=True),
#     ReduceBlock(24, 48),
#     ResidualBlock(48)
# )
#
# bn = nn.BatchNorm2d(48, momentum=0.01)
#
# aggregate_block = AggregateBlock(24, 48, 48)
#
# # conv1.load_state_dict(torch.load("conv1_weights.pt"))
# # conv2.load_state_dict(torch.load("conv2_weights.pt"))
# # before_bn.load_state_dict(torch.load("before_bn_weights.pt"))
# # after_bn.load_state_dict(torch.load("after_bn_weights.pt"))
# # reduce_block.load_state_dict(torch.load("reduce_block_weights.pt"))
# # resnet_block.load_state_dict(torch.load("resnet_block_weights.pt"))
# # bn.load_state_dict(torch.load("bn_weights.pt"))
# # residual_block.load_state_dict(torch.load("residual_block_weights.pt"))
# # aggregate_block.load_state_dict(torch.load("aggregate_block_weights.pt"))
# # aggregate_conv1.load_state_dict(torch.load("aggregate_conv1_weights.pt"))
#
#
# conv1.eval()
# conv2.eval()
# reduce_block.eval()
# resnet_block.eval()
# # bn.eval()
# residual_block.eval()
# aggregate_block.eval()
#
#
# model = Sat2Graph(batchsize=1)
# model.load_state_dict(torch.load("sat2g_tf_weights_state_dict.pt"))
# model.eval()
#
# model.model.n_4s.bn.train()
# model.model.n_8s.bn.train()
# model.model.n_16s.bn.train()
# model.model.n_32s.bn.train()
# model.model.n1_16s.bn.train()
# model.model.n2_8s.bn.train()
# model.model.n3_4s.bn.train()
#
#
# layer_result = []
# for index in range(len(data)):
#
#     input_sat = torch.from_numpy(data[index]["input_sat"]).to(torch.float32)
#     input_sat = torch.permute(input_sat, (0, 3, 1, 2))
#
#     input1 = conv2(input_sat)
#
#     input2 = residual_block(input_sat)
#     input2 = bn(input2)
#     input2 = F.relu(input2)
#
#     py_output = model(input_sat)
#     # tf_output = torch.from_numpy(data[index]["intermediate_layers"]["a1_2s_output"])
#     tf_output = torch.from_numpy(data[index]["imagegraph_output"])
#
#     tf_output = torch.permute(tf_output, (0, 3, 1, 2))
#     diff = torch.abs(py_output-tf_output)
#     layer_result.append(torch.amax(diff))
#
# print(layer_result)


# index = 0
'''supervised_loss analysis'''
# imagegraph_output = torch.from_numpy(data[index]["imagegraph_output"])
# gt_prob = torch.from_numpy(data[index]["gt_prob"])
# gt_vector = torch.from_numpy(data[index]["gt_vector"])
# gt_seg = torch.from_numpy(data[index]["gt_seg"])
# input_seg_gt_target = torch.from_numpy(data[index]["input_seg_gt_target"])
#
# imagegraph_output = torch.permute(imagegraph_output, (0, 3, 1, 2))
# gt_prob = torch.permute(gt_prob, (0, 3, 1, 2))
# gt_vector = torch.permute(gt_vector, (0, 3, 1, 2))
# gt_seg = torch.permute(gt_seg, (0, 3, 1, 2))
# input_seg_gt_target = torch.permute(input_seg_gt_target, (0, 3, 1, 2))
#
# loss_dict = data[index]["loss"]
#
# keypoint_prob_loss = loss_dict["keypoint_prob_loss"]
# direction_vector_loss = loss_dict["direction_vector_loss"]
# direction_prob_loss = loss_dict["direction_prob_loss"]
# seg_loss = loss_dict["seg_loss"]
# total = loss_dict["total"]
#
# kp_prob_loss, dir_prob_loss, dir_vector_loss, s_loss = model.supervised_loss(imagegraph_output, gt_prob, gt_vector, input_seg_gt_target)
#
# total_loss = kp_prob_loss + dir_prob_loss + dir_vector_loss + s_loss
# print(torch.abs(keypoint_prob_loss-kp_prob_loss))
# print(torch.abs(direction_prob_loss-dir_prob_loss))
# print(torch.abs(direction_vector_loss-dir_vector_loss))
# print(torch.abs(seg_loss-s_loss))
#
# print(torch.abs(total-total_loss))


'''softmax analysis'''
# softmax_diff = []
# for index in range(len(data)):
#
#     imagegraph_output = torch.from_numpy(data[index]["imagegraph_output"])
#     output_softmax = torch.from_numpy(data[index]["output_softmax"])
#
#     imagegraph_output = torch.permute(imagegraph_output, (0, 3, 1, 2))
#     output_softmax = torch.permute(output_softmax, (0, 3, 1, 2))
#
#     softmax = model.softmax_output(imagegraph_output)
#
#     diff = torch.abs(output_softmax-softmax)
#
#     softmax_diff.append(torch.amax(diff))
#
# print(softmax_diff)

