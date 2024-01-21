import torch
import scipy
import numpy as np
import random
import pickle
import json
from scipy import ndimage
import math
from PIL import Image

image_size = 256
vector_norm = 25.0


# returns numpy arrays
def rotate(sat_img, gt_seg, neighbors, samplepoints, angle=0, size=2048):
    mask = torch.zeros(gt_seg.shape)

    mask[256:size - 256, 256:size - 256] = 1

    sat_img = scipy.ndimage.rotate(sat_img, angle, reshape=False)
    gt_seg = scipy.ndimage.rotate(gt_seg, angle, reshape=False)
    mask = scipy.ndimage.rotate(mask, angle, reshape=False)

    new_neighbors = {}
    new_samplepoints = {}

    def transfer(pos, degree):
        x = pos[0] - int(size / 2)
        y = pos[1] - int(size / 2)

        new_x = x * math.cos(math.radians(degree)) - y * math.sin(math.radians(degree))
        new_y = x * math.sin(math.radians(degree)) + y * math.cos(math.radians(degree))

        return int(new_x + int(size / 2)), int(new_y + int(size / 2))

    def inrange(pos, m):
        return pos[0] > m and pos[0] < size - 1 - m and pos[1] > m and pos[1] < size - 1 - m

    for k, n in neighbors.items():
        nk = transfer(k, angle)

        if inrange(nk, 0) is False:
            continue

        new_neighbors[nk] = []

        for nei in n:
            nn = transfer(nei, angle)
            if inrange(nn, 0):
                new_neighbors[nk].append(nn)

    for k, vs in samplepoints.items():
        new_samplepoints[k] = []

        for v in vs:
            nv = transfer(v, angle)

            if inrange(nv, 256):
                new_samplepoints[k].append(nv)

    return sat_img, gt_seg, new_neighbors, new_samplepoints, mask


def neighbor_transpos(n_in):
    n_out = {}

    for k, v in n_in.items():
        nk = (k[1], k[0])
        nv = []

        for _v in v:
            nv.append((_v[1], _v[0]))

        n_out[nk] = nv

    return n_out


class Sat2GraphDataLoader:
    def __init__(self, folder, indrange=[0, 10], imgsize=256, preload_tiles=4, max_degree=6, loadseg=False,
                 random_mask=True, testing=False, dataset_image_size=2048, transpose=False):
        self.folder = folder
        self.indrange = indrange
        self.random_mask = random_mask
        self.dataset_image_size = dataset_image_size
        self.transpose = transpose
        self.preload_tiles = preload_tiles
        self.max_degree = max_degree

        # not used
        self.num = 0
        self.loadseg = loadseg

        self.image_size = imgsize
        self.testing = testing

        global image_size
        image_size = imgsize

        # initialized in preload()
        # data for preloaded image tiles
        self.noise_mask = None
        self.tiles_input = None
        self.tiles_gt_seg = None
        self.tiles_prob = None
        self.tiles_vector = None
        self.rotmask = None
        self.samplepoints = None

        # initialized in getBatch()
        # data for batch image tiles
        self.input_sat = None
        self.gt_seg = None
        self.target_prob = None
        self.target_vector = None

        # random.seed(1)

    def loadtile(self, index):
        if index not in self.indrange:
            print("Image tile not found in list")
            return None

        ind = index
        try:
            p = self.folder + f"region_{ind}_sat.png"
            sat_img = Image.open(p)
            sat_img = torch.from_numpy(np.array(sat_img).astype(float))
        except:
            print("Unable to import satellite image")
            return None

        max_v = torch.max(sat_img) + 0.0001

        sat_img = (sat_img / max_v - 0.5) * 0.9

        sat_img = torch.reshape(sat_img, [1, self.dataset_image_size, self.dataset_image_size, 3])

        tiles_prob = torch.zeros(1, self.dataset_image_size, self.dataset_image_size, 2 * (self.max_degree + 1))
        tiles_vector = torch.zeros(1, self.dataset_image_size, self.dataset_image_size, 2 * self.max_degree)

        tiles_prob[:, :, :, 0::2] = 0
        tiles_vector[:, :, :, 1::2] = 1

        try:
            neighbors = pickle.load(open(self.folder + f"/region_{ind}_refine_gt_graph.p", 'rb'))

            if self.transpose:
                neighbors = neighbor_transpos(neighbors)

            r = 1
            i = 0

            for loc, n_locs in neighbors.items():

                if loc[0] < 16 or loc[1] < 16 or loc[0] > self.dataset_image_size - 16 or loc[1] > self.dataset_image_size - 16:
                    continue

                tiles_prob[i, loc[0], loc[1], 0] = 1
                tiles_prob[i, loc[0], loc[1], 1] = 0

                for x in range(loc[0] - r, loc[0] + r + 1):
                    for y in range(loc[1] - r, loc[1] + r + 1):
                        tiles_prob[i, x, y, 0] = 1
                        tiles_prob[i, x, y, 1] = 0

                for n_loc in n_locs:
                    if n_loc[0] < 16 or n_loc[1] < 16 or n_loc[0] > self.dataset_image_size - 16 or n_loc[1] > self.dataset_image_size - 16:
                        continue

                    d = math.atan2(n_loc[1] - loc[1], n_loc[0] - loc[0]) + math.pi

                    j = int(d / (math.pi / 3.0)) % self.max_degree

                    for x in range(loc[0] - r, loc[0] + r + 1):
                        for y in range(loc[1] - r, loc[1] + r + 1):
                            tiles_prob[i, x, y, 2 + 2 * j] = 1
                            tiles_prob[i, x, y, 2 + 2 * j + 1] = 0

                            tiles_vector[i, x, y, 2 * j] = (n_loc[0] - loc[0]) / vector_norm
                            tiles_vector[i, x, y, 2 * j + 1] = (n_loc[1] - loc[1]) / vector_norm
        except:
            pass

        sat_img = torch.permute(sat_img, (0, 3, 1, 2))
        tiles_prob = torch.permute(tiles_prob, (0, 3, 1, 2))
        tiles_vector = torch.permute(tiles_vector, (0, 3, 1, 2))

        return sat_img, tiles_prob, tiles_vector

    def preload(self):
        self.noise_mask = (torch.rand(64, 64, 3)) * 1.0 + 0.5
        self.tiles_input = torch.zeros(self.preload_tiles, self.dataset_image_size, self.dataset_image_size, 3)
        self.tiles_gt_seg = torch.zeros(self.preload_tiles, self.dataset_image_size, self.dataset_image_size, 1)
        self.tiles_prob = torch.zeros(self.preload_tiles, self.dataset_image_size, self.dataset_image_size, 2 * (self.max_degree + 1))
        self.tiles_vector = torch.zeros(self.preload_tiles, self.dataset_image_size, self.dataset_image_size, 2 * self.max_degree)

        self.tiles_prob[:, :, :, 0::2] = 0
        self.tiles_prob[:, :, :, 1::2] = 1

        self.rotmask = torch.ones(self.preload_tiles, self.dataset_image_size, self.dataset_image_size)
        self.samplepoints = []

        for i in range(self.preload_tiles):
            ind = random.choice(self.indrange)
            samplepoints = json.load(open(self.folder + f"/region_{ind}_refine_gt_graph_samplepoints.json", "r"))
            self.samplepoints.append(samplepoints)

            try:
                sat_img = Image.open(self.folder + f"/region_{ind}_sat.png")
                sat_img = np.array(sat_img)
            except:
                sat_img = Image.open(self.folder + f"/region_{ind}_sat.jpg")
                sat_img = np.array(sat_img)

            max_v = np.amax(sat_img) + 0.0001

            neighbors = pickle.load(open(self.folder + f"/region_{ind}_refine_gt_graph.p", 'rb'))

            if self.transpose:
                neighbors = neighbor_transpos(neighbors)

            gt_seg = Image.open(self.folder + f"/region_{ind}_gt.png")
            gt_seg = np.array(gt_seg)

            # todo is this line necessary? self.rotmask is already initialized to a tensor of ones
            self.rotmask[i, :, :] = torch.ones(self.dataset_image_size, self.dataset_image_size)

            if self.testing is False and random.randint(0, 5) < 4:
                angle = random.randint(0, 3) * 90 + random.randint(-30, 30)
                sat_img, gt_seg, neighbors, samplepoints, rotmask = rotate(sat_img, gt_seg, neighbors, samplepoints, angle=angle, size=self.dataset_image_size)

                rotmask = torch.from_numpy(rotmask)
                self.rotmask[i, :, :] = rotmask
                # self.samplepoints[i] = samplepoints

            sat_img = torch.from_numpy(sat_img.astype(float))
            gt_seg = torch.from_numpy(gt_seg.astype(float))

            self.tiles_input[i, :, :, :] = sat_img / max_v - 0.5
            self.tiles_gt_seg[i, :, :, 0] = gt_seg / 255.0 - 0.5

            r = 1

            for loc, n_locs in neighbors.items():

                if loc[0] < 16 or loc[1] < 16 or loc[0] > self.dataset_image_size - 16 or loc[1] > self.dataset_image_size - 16:
                    continue

                self.tiles_prob[i, loc[0], loc[1], 0] = 1
                self.tiles_prob[i, loc[0], loc[1], 1] = 0

                for x in range(loc[0] - r, loc[0] + r + 1):
                    for y in range(loc[1] - r, loc[1] + r + 1):
                        self.tiles_prob[i, x, y, 0] = 1
                        self.tiles_prob[i, x, y, 1] = 0

                for n_loc in n_locs:
                    if n_loc[0] < 16 or n_loc[1] < 16 or n_loc[0] > self.dataset_image_size - 16 or n_loc[1] > self.dataset_image_size - 16:
                        continue

                    d = math.atan2(n_loc[1] - loc[1], n_loc[0] - loc[0]) + math.pi
                    j = int(d / (math.pi / 3.0)) % self.max_degree

                    for x in range(loc[0] - r, loc[0] + r + 1):
                        for y in range(loc[1] - r, loc[1] + r + 1):
                            self.tiles_prob[i, x, y, 2 + 2 * j] = 1
                            self.tiles_prob[i, x, y, 2 + 2 * j + 1] = 0

                            self.tiles_vector[i, x, y, 2 * j] = (n_loc[0] - loc[0]) / vector_norm
                            self.tiles_vector[i, x, y, 2 * j + 1] = (n_loc[1] - loc[1]) / vector_norm

            if self.testing is False:
                self.tiles_input[i, :, :, :] = self.tiles_input[i, :, :, :] * (0.8 + 0.2 * random.random()) - (random.random() * 0.4 - 0.2)

                # making everything between -0.5 and 0.5
                self.tiles_input[i, :, :, :] = torch.clip(self.tiles_input[i, :, :, :], -0.5, 0.5)

                self.tiles_input[i, :, :, 0] = self.tiles_input[i, :, :, 0] * (0.8 + 0.2 * random.random())
                self.tiles_input[i, :, :, 1] = self.tiles_input[i, :, :, 1] * (0.8 + 0.2 * random.random())
                self.tiles_input[i, :, :, 2] = self.tiles_input[i, :, :, 2] * (0.8 + 0.2 * random.random())

    def get_batch(self, batchsize=64):
        image_size = self.image_size

        self.input_sat = torch.zeros(batchsize, image_size, image_size, 3)
        self.gt_seg = torch.zeros(batchsize, image_size, image_size, 1)
        self.target_prob = torch.zeros(batchsize, image_size, image_size, 2 * (self.max_degree + 1))
        self.target_vector = torch.zeros(batchsize, image_size, image_size, 2 * self.max_degree)

        for i in range(batchsize):
            c = 0
            while True:
                tile_id = random.randint(0, self.preload_tiles-1)
                coin = random.randint(0, 99)

                if coin < 20:
                    while True:
                        x = random.randint(256, self.dataset_image_size-256-image_size)
                        y = random.randint(256, self.dataset_image_size-256-image_size)

                        if self.rotmask[tile_id, x, y] > 0.5:
                            break

                elif coin < 40:
                    sps = self.samplepoints[tile_id]['complicated_intersections']

                    if len(sps) == 0:
                        c += 1
                        continue

                    index = random.randint(0, len(sps)-1)

                    x = sps[index][0] - int(image_size/2)
                    y = sps[index][1] - int(image_size/2)

                    x = np.clip(x, 256, self.dataset_image_size - 256 - image_size)
                    y = np.clip(y, 256, self.dataset_image_size - 256 - image_size)

                elif coin < 60:
                    sps = self.samplepoints[tile_id]['parallel_road']

                    if len(sps) == 0:
                        c += 1
                        continue

                    index = random.randint(0, len(sps) - 1)

                    x = sps[index][0] - int(image_size / 2)
                    y = sps[index][1] - int(image_size / 2)

                    x = np.clip(x, 256, self.dataset_image_size - 256 - image_size)
                    y = np.clip(y, 256, self.dataset_image_size - 256 - image_size)

                else:  # overpass
                    sps = self.samplepoints[tile_id]['overpass']

                    if len(sps) == 0:
                        c += 1
                        continue

                    index = random.randint(0, len(sps) - 1)

                    x = sps[index][0] - int(image_size / 2)
                    y = sps[index][1] - int(image_size / 2)

                    x = np.clip(x, 256, self.dataset_image_size - 256 - image_size)
                    y = np.clip(y, 256, self.dataset_image_size - 256 - image_size)

                c += 1

                if torch.sum(self.tiles_gt_seg[tile_id, x:x + image_size, y:y + image_size, :] + 0.5) < 20 * 20 and c < 10:
                    continue

                self.input_sat[i, :, :, :] = self.tiles_input[tile_id, x:x + image_size, y:y + image_size, :]

                if random.randint(0, 100) < 50 and self.random_mask is True:
                    for it in range(random.randint(1, 5)):
                        xx = random.randint(0, image_size - 64 - 1)
                        yy = random.randint(0, image_size - 64 - 1)

                        self.input_sat[i, xx:xx + 64, yy:yy + 64, :] = torch.multiply(self.input_sat[i, xx:xx + 64, yy:yy + 64, :] + 0.5, self.noise_mask) - 0.5

                    for it in range(random.randint(1, 3)):
                        xx = random.randint(0, image_size - 64 - 1)
                        yy = random.randint(0, image_size - 64 - 1)

                        self.input_sat[i, xx:xx + 64, yy:yy + 64, :] = (self.noise_mask - 1.0)

                self.target_prob[i, :, :, :] = self.tiles_prob[tile_id, x:x + image_size, y:y + image_size, :]
                self.target_vector[i, :, :, :] = self.tiles_vector[tile_id, x:x + image_size, y:y + image_size, :]
                self.gt_seg[i, :, :, :] = self.tiles_gt_seg[tile_id, x:x + image_size, y:y + image_size, :]

                break

        self.input_sat = torch.permute(self.input_sat, (0, 3, 1, 2))
        self.target_prob = torch.permute(self.target_prob, (0, 3, 1, 2))
        self.target_vector = torch.permute(self.target_vector, (0, 3, 1, 2))
        self.gt_seg = torch.permute(self.gt_seg, (0, 3, 1, 2))

        return self.input_sat, self.target_prob, self.target_vector, self.gt_seg
