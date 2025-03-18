import torch
from model import Sat2Graph
from dataloader import Sat2GraphDataLoader
import os
from time import time
from PIL import Image
import numpy as np
from decoder import decode_and_vis
import sys
from douglasPeucker import draw_graph

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# path = "sat2g_tf_weights_state_dict.pt"
path = "/export/home/ayang/pytorch/run7_pt_352_8_channel12/model300000.pt"

model = Sat2Graph()
model.to(device)
model.load_state_dict(torch.load(path))
model.eval()

# model.model.n_4s.bn.train()
# model.model.n_8s.bn.train()
# model.model.n_16s.bn.train()
# model.model.n_32s.bn.train()
# model.model.n1_16s.bn.train()
# model.model.n2_8s.bn.train()
# model.model.n3_4s.bn.train()

image_size = 352
max_degree = 6

# path = "/Users/andrewyang/PycharmProjects/Sat2Graph/Sat2Graph-Server/data/20cities/"
path = "/export/home/ayang/20cities/"
tiles = [8, 9, 19, 28, 29, 39, 48, 49, 59, 68, 69, 79, 88, 89, 99, 108, 109, 119, 128, 129, 139, 148, 149, 159, 168, 169, 179]
dataloader = Sat2GraphDataLoader(path, tiles, imgsize=image_size, preload_tiles=1, random_mask=False)

os.makedirs("rawoutputs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

with torch.no_grad():
    for tile_id in tiles:
        t0 = time()
        input_sat, gt_prob, gt_vector = dataloader.loadtile(tile_id)

        gt_seg = torch.zeros(1, 1, image_size, image_size)

        # sets tensors to gpu
        input_sat = input_sat.to(device)
        gt_prob = gt_prob.to(device)
        gt_vector = gt_vector.to(device)
        gt_seg = gt_seg.to(device)

        x, y = 0, 0

        output = torch.zeros(2 + 4 * 6 + 2, 2048 + 64, 2048 + 64)
        mask = torch.ones(2 + 4 * 6 + 2, 2048 + 64, 2048 + 64) * 0.001
        weights = torch.ones(2 + 4 * 6 + 2, image_size, image_size) * 0.001
        weights[:, 32:image_size - 32, 32:image_size - 32] = 0.5
        weights[:, 56:image_size - 56, 56:image_size - 56] = 1.0
        weights[:, 88:image_size - 88, 88:image_size - 88] = 1.5

        output = output.to(device)
        mask = mask.to(device)
        weights = weights.to(device)

        input_sat = torch.nn.functional.pad(input_sat, (32, 32, 32, 32))
        gt_vector = torch.nn.functional.pad(gt_vector, (32, 32, 32, 32))
        gt_prob = torch.nn.functional.pad(gt_prob, (32, 32, 32, 32))

        input_sat = input_sat.to(torch.float32)
        gt_vector = gt_vector.to(torch.float32)
        gt_prob = gt_prob.to(torch.float32)

        for x in range(0, 352 * 6 - 176 - 88, int(176 / 2)):

            progress = int(x / 88)
            sys.stdout.write(f"\rProcessing Tile {tile_id} ...  " + ">>" * progress + "--" * (20 - progress))
            sys.stdout.flush()

            for y in range(0, 352 * 6 - 176 - 88, int(176 / 2)):

                _output = model(input_sat[:, :, x:x + image_size, y:y + image_size])

                _output = model.softmax_output(_output)

                mask[:, x:x + image_size, y:y + image_size] += weights

                output[:, x:x + image_size, y:y + image_size] += torch.mul(_output[0, :, :, :], weights)

        output = torch.div(output, mask)
        output = output[:, 32:2048 + 32, 32:2048 + 32]
        input_sat = input_sat[:, :, 32:2048 + 32, 32:2048 + 32]

        output_keypoints_img = (output[0, :, :] * 255.0).view(2048, 2048)
        output_keypoints_img = output_keypoints_img.cpu()
        output_keypoints_img = output_keypoints_img.detach().numpy().astype(np.uint8)

        Image.fromarray(output_keypoints_img).save(f"outputs/region_{tile_id}_output_keypoints.png")

        input_sat_img = ((input_sat[0, :, :, :] + 0.5) * 255.0).view((3, 2048, 2048))
        input_sat_img = input_sat_img.cpu()
        input_sat_img = torch.permute(input_sat_img, (1, 2, 0))
        input_sat_img = input_sat_img.detach().numpy()

        # input_sat_img = input_sat_img.detach().numpy().astype(np.uint8)
        # Image.fromarray(input_sat_img).save(f"outputs/region_{tile_id}_input.png")

        graph = decode_and_vis(output, f"outputs/region_{tile_id}_output", thr=0.01, edge_thr=0.05, snap=True, imagesize=2048)

        input_sat_img = draw_graph(graph, input_sat_img)
        input_sat_img = input_sat_img.astype(np.uint8)

        Image.fromarray(input_sat_img).save(f"outputs/region_{tile_id}_graph.png")

        print(" done!  time: %.2f seconds" % (time() - t0))
