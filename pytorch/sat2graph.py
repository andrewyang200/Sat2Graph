import torch
from model import Sat2Graph
from decoder import decode_and_vis
from douglasPeucker import simplify_graph

from time import time

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

path = "sat2g_tf_weights_state_dict.pt"

# todo initialize weights or not
model = Sat2Graph(image_size=352, resnet_step=8, batchsize=1, channel=12)
model.load_state_dict(torch.load(path))
model.to(device)
model.eval()


def run_sat2graph(sat_img, data):
    print('Running sat2graph...')
    sat_img = torch.from_numpy(sat_img)
    sat_img = sat_img.view((704, 704))

    print('+ Generating road graph...')
    graph = generate_graph(sat_img, data)

    print('+ fitting road graph to bounding box...')
    lines, points = fit_to_bbox(graph)

    print('Sat2graph done!')
    return lines, points


# todo understand what this code is doing
def fit_to_bbox(graph):
    lines = []
    points = []

    for nid, nei in graph.iteritems():
        for nn in nei:
            if in_range(nn) or in_range(nid):
                edge = (add_bias(nid), add_bias(nn))
                edge_ = (add_bias(nn), add_bias(nid))
                if edge not in lines and edge_ not in lines:
                    lines.append(edge)

        if in_range(nid) and len(nei) != 2:
            points.append(add_bias(nid))

    return lines, points


def generate_graph(sat_img, data):
    with torch.no_grad():
        v_thr = data["v_thr"]
        e_thr = data["e_thr"]
        snap_dist = data["snap_dist"]
        snap_w = data["snap_w"]

        max_v = 255
        sat_img = (sat_img / max_v - 0.5) * 0.9
        sat_img = sat_img.view((1, 3, 704, 704))

        sat_img.to(device)

        # todo should image size still be 352
        image_size = 352

        weights = torch.ones(2 + 4 * 6 + 2, image_size, image_size) * 0.001
        weights[:, 32:image_size - 32, 32:image_size - 32] = 0.5
        weights[:, 56:image_size - 56, 56:image_size - 56] = 1.0
        weights[:, 88:image_size - 88, 88:image_size - 88] = 1.5

        mask = torch.zeros((2 + 4 * 6 + 2, 704, 704))
        output = torch.zeros((2 + 4 * 6 + 2, 704, 704))

        output = output.to(device)
        mask = mask.to(device)
        weights = weights.to(device)

        t0 = time()
        for x in range(0, 704 - 176 - 88, int(176 / 2)):
            for y in range(0, 704 - 176 - 88, int(176 / 2)):
                _output = model(sat_img[:, :, x:x + image_size, y:y + image_size])

                _output = model.softmax_output(_output)

                mask[:, x:x + image_size, y:y + image_size] += weights

                output[:, x:x + image_size, y:y + image_size] += torch.mul(_output[0, :, :, :], weights)

        print("GPU time:", time() - t0)
        t0 = time()

        output = torch.div(output, mask)
        output_file = 'TEST01_'

        graph = decode_and_vis(output, output_file, thr=v_thr, edge_thr=e_thr, angledistance_weight=snap_w, snap_dist=snap_dist, snap=True, imagesize=704)

        print("Decode time:", time() - t0)
        t0 = time()

        graph = simplify_graph(graph)

        print("Graph simplify time:", time() - t0)
        return graph


def add_bias(loc):
    bias_x = -102
    bias_y = -102
    return loc[0] + bias_x, loc[1] + bias_y


def in_range(loc):
    if loc[0] > 102 and loc[0] < 602 and loc[1] > 102 and loc[1] < 602:
        return True
    else:
        return False
