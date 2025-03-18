from PIL import Image
import numpy as np
import torch
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import pickle
from torch.utils.tensorboard import SummaryWriter
from decoder import decode_and_vis


"""This script compared the differences in node dictionaries of tensorflow output graph vs pytorch output graph"""


# write = SummaryWriter("folder")

# py = pickle.load(open("/Users/andrewyang/PycharmProjects/pytorch/outputs/region_8_output_graph_bn.p", "rb")
# tf = pickle.load(open("/Users/andrewyang/PycharmProjects/Sat2Graph/Sat2Graph-Server/model/outputs/region_8_output_graph.p", "rb"), encoding='latin1')

# p = "/Users/andrewyang/PycharmProjects/model_infer_tests/infer_test_samples.pickle"

# data = pickle.load(open(p, "rb"), encoding='latin1')

# '''(1557, 1613)
# (1774, 90)
# (1843, 224)'''
# for keys in py:
#     if tf.get(keys) is None:
#         print(keys)
#
# '''(15, 918)
# (2002, 2046)'''
# for keys in tf:
#     if py.get(keys) is None:
#         print(keys)
#
# print(data[0]["output_softmax"].shape)
#
# for index in range(5):
#     decode_and_vis(data[index]["output_softmax"][0], "sample%d" % index, thr=0.01, snap=True, imagesize=352)
#
# sample = pickle.load(open("/Users/andrewyang/PycharmProjects/pytorch/sample3_graph_bn.p", "rb"))
#
# test = pickle.load(open("/Users/andrewyang/PycharmProjects/model_infer_tests/region0_batch0_output_graph_0.01_snap.png_graph.p", "rb"), encoding = "latin1")
#
#
# for keys in sample:
#     if test.get(keys) is None:
#         print(keys)
#
# print('//')
# for keys in test:
#     if sample.get(keys) is None:
#         print(keys)

# for keys in test:
#    if set(sample[keys]) != set(test[keys]):
#        print(keys)
