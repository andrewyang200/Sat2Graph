import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ResidualBlock
from layers import ConvLayer
from layers import TFConvLayer

MAX_DEGREE = 6


class Sat2Graph(nn.Module):
    def __init__(self, image_size=352, image_ch=3, batchsize=8, resnet_step=8, channel=12, joint_with_seg=True, initialize=False):
        super(Sat2Graph, self).__init__()
        self.image_size = image_size
        self.image_ch = image_ch
        self.batchsize = batchsize
        self.resnet_step = resnet_step
        self.channel = channel
        self.joint_with_seg = joint_with_seg

        self.model = DeepLayerAggregationNetWithResnet(image_ch, 2+MAX_DEGREE*4+(2 if joint_with_seg is True else 0), channel, resnet_step, initialize=initialize)
        num_unet = len(list(filter(lambda p: p.requires_grad, self.model.parameters())))
        print("Number of Weights:", num_unet)

    def forward(self, x):
        output = self.model(x)
        return output

    def unstack(self, tensor, axis=1, size=None):
        ts = torch.unbind(tensor, dim=axis)
        new_ts = []

        for t in ts:
            if size is None:
                new_ts.append(t.view(-1, 1, self.image_size, self.image_size))
            else:
                new_ts.append(t.view(-1, 1, size, size))

        return new_ts

    # changed self.input_seg_gt_target to input_seg_gt_target and made it an input parameter
    def supervised_loss(self, imagegraph_output, imagegraph_target_prob, imagegraph_target_vector, input_seg_gt_target):
        imagegraph_outputs = self.unstack(imagegraph_output, axis=1)
        imagegraph_target_probs = self.unstack(imagegraph_target_prob, axis=1)
        imagegraph_target_vectors = self.unstack(imagegraph_target_vector, axis=1)

        soft_mask = torch.clamp(imagegraph_target_probs[0] - 0.01, min=0.0, max=0.99)
        soft_mask = soft_mask+0.01

        soft_mask2 = soft_mask.view(self.batchsize, self.image_size, self.image_size)

        keypoint_prob_output = torch.cat(imagegraph_outputs[0:2], dim=1)
        keypoint_prob_target = torch.cat(imagegraph_target_probs[0:2], dim=1)

        keypoint_prob_loss = torch.mean(F.cross_entropy(keypoint_prob_output, keypoint_prob_target))

        direction_prob_loss = 0

        for i in range(MAX_DEGREE):
            prob_output = torch.cat(imagegraph_outputs[2 + i * 4: 2 + i * 4 + 2], dim=1)
            prob_target = torch.cat(imagegraph_target_probs[2 + i * 2: 2 + i * 2 + 2], dim=1)

            direction_prob_loss += torch.mean(soft_mask2 * F.cross_entropy(prob_output, prob_target, reduction='none'))

        direction_prob_loss /= MAX_DEGREE

        direction_vector_loss = 0

        for i in range(MAX_DEGREE):
            vector_output = torch.cat(imagegraph_outputs[2 + i * 4 + 2: 2 + i * 4 + 4], dim=1)
            vector_target = torch.cat(imagegraph_target_vectors[i * 2:i * 2 + 2], dim=1)

            direction_vector_loss += torch.mean(soft_mask * ((vector_output - vector_target) ** 2))

        direction_vector_loss /= MAX_DEGREE

        if self.joint_with_seg:
            seg_loss = torch.mean(F.cross_entropy(torch.cat([imagegraph_outputs[2 + MAX_DEGREE * 4], imagegraph_outputs[2 + MAX_DEGREE * 4 + 1]], dim=1), input_seg_gt_target))

            return keypoint_prob_loss, direction_prob_loss * 10.0, direction_vector_loss * 1000.0, seg_loss * 0.1

        else:

            return keypoint_prob_loss, direction_prob_loss * 10.0, direction_vector_loss * 1000.0, keypoint_prob_loss - keypoint_prob_loss

    def softmax_output(self, imagegraph_output):
        imagegraph_outputs = self.unstack(imagegraph_output, axis=1)

        new_outputs = []

        new_outputs.append(torch.sigmoid(imagegraph_outputs[0] - imagegraph_outputs[1]))  # vertex probability
        new_outputs.append(1.0 - new_outputs[-1])  # binary encoding

        for i in range(MAX_DEGREE):
            # Applies sigmoid to the corresponding image outputs and adds to the list
            new_outputs.append(torch.sigmoid(imagegraph_outputs[2 + i * 4] - imagegraph_outputs[2 + i * 4 + 1]))  # edge probability
            new_outputs.append(1.0 - new_outputs[-1])
            new_outputs.append(torch.cat(imagegraph_outputs[2 + i * 4 + 2:2 + i * 4 + 4], dim=1))  # keep the edge directions the same

        if self.joint_with_seg:
            # Applies sigmoid to the last image outputs and adds to the list
            new_outputs.append(torch.sigmoid(
                imagegraph_outputs[2 + 4 * MAX_DEGREE] - imagegraph_outputs[2 + 4 * MAX_DEGREE + 1]))
            new_outputs.append(1.0 - new_outputs[-1])

        return torch.cat(new_outputs, dim=1)

    def get_jointwithseg(self):
        return self.joint_with_seg


class ReduceBlock(nn.Module):
    def __init__(self, in_ch, out_ch, initialize=False):
        super(ReduceBlock, self).__init__()
        self.conv1 = ConvLayer(in_ch, in_ch, kernel_size=3, stride=1, batchnorm=True, initialize=initialize)
        self.conv2 = ConvLayer(in_ch, out_ch, kernel_size=3, stride=2, batchnorm=True, initialize=initialize)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, ch, resnet_step=0, initialize=False):
        super(ResnetBlock, self).__init__()
        self.resnet_step = resnet_step
        self.resnet = nn.ModuleList()
        for i in range(resnet_step):
            self.resnet.append(ResidualBlock(ch, downsample=False, initialize=initialize))

        self.bn = nn.BatchNorm2d(ch, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.resnet_step > 0:
            for i in range(self.resnet_step):
                x = self.resnet[i](x)

        x = self.bn(x)
        x = self.relu(x)

        return x


class AggregateBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, batchnorm=True, initialize=False):
        super(AggregateBlock, self).__init__()
        self.conv1 = ConvLayer(in_ch2, in_ch2, kernel_size=3, stride=2, batchnorm=batchnorm, deconv=True, initialize=initialize)
        # self.conv1 = TFConvLayer(in_ch2, in_ch2, kernel_size=3, stride=2, batchnorm=batchnorm, deconv=True, initialize=initialize)
        self.conv2 = ConvLayer(in_ch1+in_ch2, in_ch1+in_ch2, kernel_size=3, stride=1, batchnorm=batchnorm, initialize=initialize)
        self.conv3 = ConvLayer(in_ch1+in_ch2, out_ch, kernel_size=3, stride=1, batchnorm=batchnorm, initialize=initialize)

    def forward(self, x1, x2):
        x2 = self.conv1(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DeepLayerAggregationNetWithResnet(nn.Module):
    def __init__(self, in_ch, out_ch, ch, resnet_step, initialize=False):
        super(DeepLayerAggregationNetWithResnet, self).__init__()
        self.conv1 = ConvLayer(in_ch, ch, kernel_size=5, stride=1, batchnorm=False, initialize=initialize)
        self.conv2 = ConvLayer(ch, ch*2, kernel_size=5, stride=2, batchnorm=True, initialize=initialize)

        self.r_4s = ReduceBlock(ch*2, ch*4, initialize=initialize)
        self.n_4s = ResnetBlock(ch*4, resnet_step=int(resnet_step/8), initialize=initialize)

        self.r_8s = ReduceBlock(ch*4, ch*8, initialize=initialize)
        self.n_8s = ResnetBlock(ch*8, resnet_step=int(resnet_step/4), initialize=initialize)

        self.r_16s = ReduceBlock(ch*8, ch*16, initialize=initialize)
        self.n_16s = ResnetBlock(ch*16, resnet_step=int(resnet_step/2), initialize=initialize)

        self.r_32s = ReduceBlock(ch*16, ch*32, initialize=initialize)
        self.n_32s = ResnetBlock(ch*32, resnet_step=resnet_step, initialize=initialize)

        self.a1_2s = AggregateBlock(ch*2, ch*4, ch*4, initialize=initialize)
        self.a1_4s = AggregateBlock(ch*4, ch*8, ch*8, initialize=initialize)
        self.a1_8s = AggregateBlock(ch*8, ch*16, ch*16, initialize=initialize)
        self.a1_16s = AggregateBlock(ch*16, ch*32, ch*32, initialize=initialize)
        self.n1_16s = ResnetBlock(ch*32, resnet_step=int(resnet_step/2), initialize=initialize)

        self.a2_2s = AggregateBlock(ch*4, ch*8, ch*4, initialize=initialize)
        self.a2_4s = AggregateBlock(ch*8, ch*16, ch*8, initialize=initialize)
        self.a2_8s = AggregateBlock(ch*16, ch*32, ch*16, initialize=initialize)
        self.n2_8s = ResnetBlock(ch*16, resnet_step=int(resnet_step/4), initialize=initialize)

        self.a3_2s = AggregateBlock(ch*4, ch*8, ch*4, initialize=initialize)
        self.a3_4s = AggregateBlock(ch*8, ch*16, ch*8, initialize=initialize)
        self.n3_4s = ResnetBlock(ch*8, resnet_step=int(resnet_step/8), initialize=initialize)

        self.a4_2s = AggregateBlock(ch*4, ch*8, ch*8, initialize=initialize)

        self.conv3 = ConvLayer(ch*8, ch*4, kernel_size=3, stride=1, batchnorm=True, initialize=initialize)

        self.a5_2s = AggregateBlock(ch, ch*4, ch*4, batchnorm=False, initialize=initialize)

        self.conv4 = ConvLayer(ch*4, out_ch, kernel_size=3, stride=1, batchnorm=False, activation="linear", initialize=initialize)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        r_4s = self.r_4s(conv2)
        n_4s = self.n_4s(r_4s)

        r_8s = self.r_8s(n_4s)
        n_8s = self.n_8s(r_8s)

        r_16s = self.r_16s(n_8s)
        n_16s = self.n_16s(r_16s)

        r_32s = self.r_32s(n_16s)
        n_32s = self.n_32s(r_32s)

        a1_2s = self.a1_2s(conv2, n_4s)
        a1_4s = self.a1_4s(n_4s, n_8s)
        a1_8s = self.a1_8s(n_8s, n_16s)
        a1_16s = self.a1_16s(n_16s, n_32s)
        n1_16s = self.n1_16s(a1_16s)

        a2_2s = self.a2_2s(a1_2s, a1_4s)
        a2_4s = self.a2_4s(a1_4s, a1_8s)
        a2_8s = self.a2_8s(a1_8s, n1_16s)
        n2_8s = self.n2_8s(a2_8s)

        a3_2s = self.a3_2s(a2_2s, a2_4s)
        a3_4s = self.a3_4s(a2_4s, n2_8s)
        n3_4s = self.n3_4s(a3_4s)

        a4_2s = self.a4_2s(a3_2s, n3_4s)

        conv3 = self.conv3(a4_2s)

        a5_2s = self.a5_2s(conv1, conv3)

        out = self.conv4(a5_2s)

        return out
