from model import Sat2Graph
from dataloader import Sat2GraphDataLoader
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from time import time
from decoder import decode_and_vis
from PIL import Image
import sys
import argparse

max_degree = 6

parser = argparse.ArgumentParser()

parser.add_argument('-model_save', action='store', dest='model_save', type=str,
                    help='model save folder', required=True)

parser.add_argument('-instance_id', action='store', dest='instance_id', type=str,
                    help='instance_id', required=True)

parser.add_argument('-model_recover', action='store', dest='model_recover', type=str,
                    help='saved model path', required=False, default=None)

parser.add_argument('-image_size', action='store', dest='image_size', type=int,
                    help='image size', required=False, default=352)

parser.add_argument('-lr', action='store', dest='lr', type=float,
                    help='learning rate', required=False, default=0.001)

parser.add_argument('-lr_decay', action='store', dest='lr_decay', type=float,
                    help='learning rate decay', required=False, default=0.5)

parser.add_argument('-lr_decay_step', action='store', dest='lr_decay_step', type=int,
                    help='learning rate decay step', required=False, default=50000)

parser.add_argument('-weight_decay', action='store', dest='weight_decay', type=float,
                    help='regularizer', required=False, default=0.0)

parser.add_argument('-init_step', action='store', dest='init_step', type=int,
                    help='initial step size', required=False, default=0)

parser.add_argument('-resnet_step', action='store', dest='resnet_step', type=int,
                    help='resnet step', required=False, default=8)

parser.add_argument('-spacenet', action='store', dest='spacenet', type=str,
                    help='spacenet folder', required=False, default="")

parser.add_argument('-channel', action='store', dest='channel', type=int,
                    help='channel size', required=False, default=12)

parser.add_argument('-mode', action='store', dest='mode', type=str,
                    help='mode [train][test][validate]', required=False, default="train")

parser.add_argument('-replicate', action='store', dest='replicate', type=bool,
                    help='replicate tensorflow model', required=False, default=True)

parser.add_argument('-initialize', action='store', dest='initialize', type=bool,
                    help='weight and bias initializer', required=False, default=False)

parser.add_argument('-cuda', action='store', dest='cuda', type=int,
                    help='cuda device number', required=False, default=0)

args = parser.parse_args()

# dataset_path = "/Users/andrewyang/PycharmProjects/Sat2Graph/Sat2Graph-Server/data/20cities/"  # update
dataset_path = '/export/home/ayang/20cities'

os.makedirs("tensorboard", exist_ok=True)
writer = SummaryWriter("tensorboard/" + args.model_save)


instance_id = args.model_save + "_" + args.instance_id + "_" + str(args.image_size) + "_" + str(args.resnet_step) + f"_channel{args.channel}"
image_size = args.image_size

if args.mode != "train":
    batch_size = 1
else:
    batch_size = 2

validation_folder = "validation_" + instance_id
model_save_folder = instance_id

os.makedirs(validation_folder, exist_ok=True)
os.makedirs(model_save_folder, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

model = Sat2Graph(image_size=image_size, resnet_step=args.resnet_step, batchsize=batch_size, channel=args.channel, initialize=args.initialize)
model.to(device)

if args.model_recover is not None:
    print("model recover", args.model_recover)
    model.load_state_dict(torch.load(args.model_recover))

if args.spacenet == "":
    print("Using 20-city dataset")

    indrange_train = []
    indrange_test = []
    indrange_validation = []

    for x in range(180):
        if x % 10 < 8:
            indrange_train.append(x)

        if x % 10 == 9:
            indrange_test.append(x)

        if x % 20 == 18:
            indrange_validation.append(x)

        if x % 20 == 8:
            indrange_test.append(x)

    print("training set:", indrange_train)
    print("testing set:", indrange_test)
    print("validation set:", indrange_validation)

    if args.mode == "train":
        dataloader_train = Sat2GraphDataLoader(dataset_path, indrange_train, imgsize=image_size, preload_tiles=4, testing=False, random_mask=True)
        dataloader_train.preload()

        dataloader_test = Sat2GraphDataLoader(dataset_path, indrange_validation, imgsize=image_size, preload_tiles=len(indrange_validation), random_mask=False, testing=True)
        dataloader_test.preload()
    else:
        pass
else:
    pass

validation_data = []
test_size = 32

for j in range(int(test_size/batch_size)):
    input_sat, gt_prob, gt_vector, gt_seg = dataloader_test.get_batch(batch_size)
    validation_data.append([torch.detach(input_sat), torch.detach(gt_prob), torch.detach(gt_vector), torch.detach(gt_seg)])

step = args.init_step
lr = args.lr

t_load = 0
t_last = time()
t_train = 0

test_loss = 0

sum_loss = 0
sum_prob_loss = 0
sum_vector_loss = 0
sum_seg_loss = 0

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

while True:
    t0 = time()
    input_sat, gt_prob, gt_vector, gt_seg = dataloader_train.get_batch(batch_size)
    input_seg_gt_target = torch.cat([gt_seg + 0.5, 0.5 - gt_seg], dim=1)

    input_sat = input_sat.to(device)
    gt_prob = gt_prob.to(device)
    gt_vector = gt_vector.to(device)
    gt_seg = gt_seg.to(device)
    input_seg_gt_target = input_seg_gt_target.to(device)

    t_load += time() - t0

    t0 = time()

    model.train()
    imagegraph_output = model(input_sat)

    # Calculating loss
    keypoint_prob_loss, direction_prob_loss, direction_vector_loss, seg_loss = model.supervised_loss(imagegraph_output, gt_prob, gt_vector, input_seg_gt_target)
    prob_loss = keypoint_prob_loss + direction_prob_loss
    if model.get_jointwithseg():
        loss = prob_loss + direction_vector_loss + seg_loss
    else:
        loss = prob_loss + direction_vector_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    sum_loss += loss
    sum_prob_loss += prob_loss
    sum_vector_loss += direction_vector_loss
    sum_seg_loss += seg_loss

    t_train += time() - t0

    if step % 10 == 0:
        sys.stdout.write(f"\rbatch:{step} " + ">>" * int((step - int(step / 200) * 200) / 10) + "--" * int(((int(step / 200) + 1) * 200 - step) / 10))
        sys.stdout.flush()

    if step > -1 and step % 200 == 0:
        sum_loss /= 200

        if step % 1000 == 0 or (step < 1000 and step % 200 == 0):
            test_loss = 0

            for j in range(-1, int(test_size / batch_size)):
                if j >= 0:
                    input_sat, gt_prob, gt_vector, gt_seg = validation_data[j][0], validation_data[j][1], validation_data[j][2], validation_data[j][3]
                    input_seg_gt_target = torch.cat([gt_seg + 0.5, 0.5 - gt_seg], dim=1)

                    input_sat = input_sat.to(device)
                    gt_prob = gt_prob.to(device)
                    gt_vector = gt_vector.to(device)
                    gt_seg = gt_seg.to(device)
                    input_seg_gt_target = input_seg_gt_target.to(device)

                if j == 0:
                    test_loss = 0

                # todo set model to eval and with no grad
                with torch.no_grad():
                    model.eval()

                    if args.replicate:
                        model.model.n_4s.bn.train()
                        model.model.n_8s.bn.train()
                        model.model.n_16s.bn.train()
                        model.model.n_32s.bn.train()
                        model.model.n1_16s.bn.train()
                        model.model.n2_8s.bn.train()
                        model.model.n3_4s.bn.train()

                    img_graph_output = model(input_sat)

                    output = model.softmax_output(img_graph_output)
                    output = output.to(device)

                    test_keypoint_prob_loss, test_direction_prob_loss, test_direction_vector_loss, test_seg_loss = model.supervised_loss(img_graph_output, gt_prob, gt_vector, input_seg_gt_target)
                    test_prob_loss = test_keypoint_prob_loss + test_direction_prob_loss
                    if model.get_jointwithseg():
                        _test_loss = test_prob_loss + test_direction_vector_loss + test_seg_loss
                    else:
                        _test_loss = test_prob_loss + test_direction_vector_loss

                    test_loss += _test_loss

                    if step == 1000 or step % 2000 == 0 or (step < 1000 and step % 200 == 0):
                        for k in range(batch_size):
                            # todo move these tensors back to cpu before saving as image
                            input_sat_img = ((input_sat[k, :, :, :] + 0.5) * 255.0).view((3, image_size, image_size))
                            input_sat_img = input_sat_img.cpu()
                            input_sat_img = torch.permute(input_sat_img, (1, 2, 0))
                            input_sat_img = input_sat_img.detach().numpy().astype(np.uint8)
                            Image.fromarray(input_sat_img).save(validation_folder + f"/tile{(j * batch_size + k)}_input_sat.png")

                            output_img = (output[k, -2, :, :] * 255.0).view((image_size, image_size))
                            output_img = output_img.cpu()
                            output_img = output_img.detach().numpy().astype(np.uint8)
                            Image.fromarray(output_img).save(validation_folder + f"/tile{(j * batch_size + k)}_output_seg.png")

                            gt_seg_output = ((gt_seg[k, 0, :, :] + 0.5) * 255.0).view((image_size, image_size))
                            gt_seg_output = gt_seg_output.cpu()
                            gt_seg_output = gt_seg_output.detach().numpy().astype(np.uint8)
                            Image.fromarray(gt_seg_output).save(validation_folder + f"/tile{(j * batch_size + k)}_gt_seg.png")

                            output_keypoints_img = (output[k, 0, :, :] * 255.0).view((image_size, image_size))
                            output_keypoints_img = output_keypoints_img.cpu()
                            output_keypoints_img = output_keypoints_img.detach().numpy().astype(np.uint8)
                            Image.fromarray(output_keypoints_img).save(validation_folder + f"/tile{(j * batch_size + k)}_output_keypoints.png")

                            decode_and_vis(output[k, 0:2 + 4 * max_degree, :, :].view((2 + 4 * max_degree, image_size, image_size)),
                                           validation_folder + f"/tile{(j * batch_size + k)}_output_graph_0.01_snap.png", thr=0.01, snap=True, imagesize=image_size)

            test_loss /= test_size / batch_size

        print("")
        print(f'step: {step}, loss: {sum_loss}, test_loss: {test_loss}, prob_loss: {sum_prob_loss / 200.0}, vector_loss: {sum_vector_loss / 200.0}, seg_loss: {sum_seg_loss / 200.0}')

        writer.add_scalars('Loss_Curve', {'loss': sum_loss,
                                             'test_loss': test_loss,
                                             'prob_loss': sum_prob_loss/200.0,
                                             'vector_loss': sum_vector_loss / 200.0,
                                             'seg_loss': sum_seg_loss / 200.0}, step)
        writer.flush()

        sum_prob_loss = 0
        sum_vector_loss = 0
        sum_seg_loss = 0
        sum_loss = 0

    if step > 0 and step % 400 == 0:
        dataloader_train.preload()

    if step > 0 and step % 2000 == 0:
        print(f'time elapsed: {time() - t_last}, loading time: {t_load}, training time: {t_train}')
        t_last = time()
        t_load = 0
        t_train = 0

    if step > 0 and (step % 10000 == 0):
        torch.save(model.state_dict(), model_save_folder + f"/model{step}.pt")

    step += 1

    if step == 600000 + 2:
        break

print('finished training')

writer.close()
