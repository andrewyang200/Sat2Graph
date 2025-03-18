import torch
from model import Sat2Graph
from dataloader import Sat2GraphDataLoader
import os
from time import time
from PIL import Image
import numpy as np
import pickle


"""This script does a prelim test on whether or not the inference code is producing correct output keypoints 
on a smaller sized input image"""

#
# index = 1
# p = "/Users/andrewyang/PycharmProjects/model_infer_tests/infer_test_samples.pickle"
#
# data = pickle.load(open(p, "rb"), encoding='latin1')
#
# img_id = data[index]["img_id"]
# gt_vector = torch.from_numpy(data[index]["gt_vector"])
# imagegraph_output = torch.from_numpy(data[index]["imagegraph_output"])
# gt_seg = torch.from_numpy(data[index]["gt_seg"])
# output_softmax = torch.from_numpy(data[index]["output_softmax"])
# input_sat = torch.from_numpy(data[index]["input_sat"])
# gt_prob = torch.from_numpy(data[index]["gt_prob"])
# input_seg_gt_target = torch.from_numpy(data[index]["input_seg_gt_target"])
# loss_dict = data[index]["loss"]
#
#
# gt_vector = torch.permute(gt_vector, (0, 3, 1, 2))
# imagegraph_output = torch.permute(imagegraph_output, (0, 3, 1, 2))
# gt_seg = torch.permute(gt_seg, (0, 3, 1, 2))
# output_softmax = torch.permute(output_softmax, (0, 3, 1, 2))
# input_sat = torch.permute(input_sat, (0, 3, 1, 2))
# gt_prob = torch.permute(gt_prob, (0, 3, 1, 2))
# input_seg_gt_target = torch.permute(input_seg_gt_target, (0, 3, 1, 2))
#
#
#
# print(imagegraph_output.shape)
#
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")
#
# model = Sat2Graph()
# model.load_state_dict(torch.load("sat2g_tf_weights_state_dict.pt"))
# model.eval()
# model.to(device)
#
# image_size = 352
# max_degree = 6
#
# os.makedirs("rawoutputs", exist_ok=True)
# os.makedirs("outputs", exist_ok=True)
#
# with torch.no_grad():
#     t0 = time()
#
#
#     # included a batch number flag
#     gt_imagegraph = torch.zeros(1, 26, 352, 352)
#
#     # sets tensors to gpu
#     input_sat = input_sat.to(device)
#     gt_prob = gt_prob.to(device)
#     gt_vector = gt_vector.to(device)
#     gt_seg = gt_seg.to(device)
#     gt_imagegraph = gt_imagegraph.to(device)
#
#     gt_imagegraph[:, 0:2, :, :] = gt_prob[0, 0:2, :, :]
#
#     for k in range(max_degree):
#         gt_imagegraph[:, 2 + k * 4:2 + k * 4 + 2, :, :] = gt_prob[0, 2 + k * 2:2 + k * 2 + 2, :, :]
#         gt_imagegraph[:, 2 + k * 4 + 2:2 + k * 4 + 4, :, :] = gt_vector[0, k * 2:k * 2 + 2, :, :]
#
#     input_sat = input_sat.to(torch.float32)
#     gt_vector = gt_vector.to(torch.float32)
#     gt_prob = gt_prob.to(torch.float32)
#
#     _output = model(input_sat)
#
#     _output = _output.to(device)
#
#     _output = model.softmax_output(_output)
#
#     print(_output.shape)
#
#     output_keypoints_img = (_output[0, 0, :, :] * 255.0).view(352, 352)
#     output_keypoints_img = output_keypoints_img.cpu()
#     output_keypoints_img = output_keypoints_img.detach().numpy().astype(np.uint8)
#
#     Image.fromarray(output_keypoints_img).save("outputs/sample_region_%d_output_keypoints.png" % img_id)
#
#     # with open('result.pickle', 'wb') as f:
#     #     pickle.dump(_output, f)
#
#
#     input_sat_img = ((input_sat[0, :, :, :] + 0.5) * 255.0).view((3, 352, 352))
#     input_sat_img = input_sat_img.cpu()
#     input_sat_img = torch.permute(input_sat_img, (1, 2, 0))
#     input_sat_img = input_sat_img.detach().numpy().astype(np.uint8)
#
#     Image.fromarray(input_sat_img).save("outputs/sample_region_%d_input.png" % img_id)
#
#     result = pickle.load(open('result.pickle', 'rb'))
#
